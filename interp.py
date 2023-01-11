import torch

EPSILON = 0.0000000001

def _pairwise_squared_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance among a (batch) matrix's rows (2nd dim).
    This saves a bit of computation vs. using
    `_cross_squared_distance_matrix(x, x)`
    Args:
      x: `[batch_size, n, d]` float `Tensor`.
    Returns:
      squared_dists: `[batch_size, n, n]` float `Tensor`, where
      `squared_dists[b,i,j] = ||x[b,i,:] - x[b,j,:]||^2`.
    """
    x_x_transpose = x@x.swapaxes(-1, -2)
    # x_norm_squared = torch.diag(x_x_transpose)
    x_norm_squared = torch.diagonal(x_x_transpose, 0, -1, -2)
    x_norm_squared_tile = torch.unsqueeze(x_norm_squared, 2)
    # squared_dists[b,i,j] = ||x_bi - x_bj||^2 =
    # = x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = (
        x_norm_squared_tile
        - 2 * x_x_transpose
        + torch.swapaxes(x_norm_squared_tile, 1, 2)
    )

    return squared_dists


def _cross_squared_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance between two (batch) matrices' rows (2nd dim).
    Computes the pairwise distances between rows of x and rows of y.
    Args:
      x: `[batch_size, n, d]` float `Tensor`.
      y: `[batch_size, m, d]` float `Tensor`.
    Returns:
      squared_dists: `[batch_size, n, m]` float `Tensor`, where
      `squared_dists[b,i,j] = ||x[b,i,:] - y[b,j,:]||^2`.
    """
    x_norm_squared = torch.sum(torch.square(x), 2)
    y_norm_squared = torch.sum(torch.square(y), 2)

    # Expand so that we can broadcast.
    x_norm_squared_tile = torch.unsqueeze(x_norm_squared, 2)
    y_norm_squared_tile = torch.unsqueeze(y_norm_squared, 1)

    x_y_transpose = x@y.swapaxes(-1, -2)

    # squared_dists[b,i,j] = ||x_bi - y_bj||^2 =
    # x_bi'x_bi- 2x_bi'x_bj + x_bj'x_bj
    squared_dists = x_norm_squared_tile - 2 * x_y_transpose + y_norm_squared_tile

    return squared_dists

def _phi(r: torch.Tensor, order: int) -> torch.Tensor:
    """Coordinate-wise nonlinearity used to define the order of the
    interpolation.
    See https://en.wikipedia.org/wiki/Polyharmonic_spline for the definition.
    Args:
      r: input op.
      order: interpolation order.
    Returns:
      `phi_k` evaluated coordinate-wise on `r`, for `k = r`.
    """

    # using EPSILON prevents log(0), sqrt0), etc.
    # sqrt(0) is well-defined, but its gradient is not
    if order == 1:
        r = torch.maximum(r, r.new([EPSILON]))
        r = torch.sqrt(r)
        return r
    elif order == 2:
        return 0.5 * r * torch.log(torch.maximum(r, r.new([EPSILON])))
    elif order == 4:
        return 0.5 * torch.square(r) * torch.log(torch.maximum(r, r.new([EPSILON])))
    elif order % 2 == 0:
        r = torch.maximum(r, r.new([EPSILON]))
        return 0.5 * torch.pow(r, 0.5 * order) * torch.log(r)
    else:
        r = torch.maximum(r, r.new([EPSILON]))
        return torch.pow(r, 0.5 * order)


def _solve_interpolation(
    train_points: torch.Tensor,
    train_values: torch.Tensor,
    order: int,
    regularization_weight: torch.Tensor,
) -> torch.Tensor:
    r"""Solve for interpolation coefficients.
    Computes the coefficients of the polyharmonic interpolant for the
    'training' data defined by `(train_points, train_values)` using the kernel
    $\phi$.
    Args:
      train_points: `[b, n, d]` interpolation centers.
      train_values: `[b, n, k]` function values.
      order: order of the interpolation.
      regularization_weight: weight to place on smoothness regularization term.
    Returns:
      w: `[b, n, k]` weights on each interpolation center
      v: `[b, d, k]` weights on each input dimension
    Raises:
      ValueError: if d or k is not fully specified.
    """

    # These dimensions are set dynamically at runtime.
    if len(train_points.shape) != 3:
        raise ValueError(
            " input points shape must be [b, n, d]"
        )
    if len(train_values.shape) != 3:
        raise ValueError(
            " output values shape must be [b, n, k]"
        )

    b, n, d = train_points.shape

    if d is None:
        raise ValueError(
            "The dimensionality of the input points (d) must be "
            "statically-inferrable."
        )

    k = train_values.shape[-1]
    if k is None:
        raise ValueError(
            "The dimensionality of the output values (k) must be "
            "statically-inferrable."
        )

    # First, rename variables so that the notation (c, f, w, v, A, B, etc.)
    # follows https://en.wikipedia.org/wiki/Polyharmonic_spline.
    # To account for python style guidelines we use
    # matrix_a for A and matrix_b for B.

    c = train_points
    f = train_values

    # Next, construct the linear system.

    matrix_a = _phi(_pairwise_squared_distance_matrix(c), order)  # [b, n, n]
    if regularization_weight > 0:
        batch_identity_matrix = torch.unsqueeze(torch.eye(n, dtype=c.dtype, device=c.device), 0)
        matrix_a += regularization_weight * batch_identity_matrix

    # Append ones to the feature values for the bias term
    # in the linear model.
    ones = torch.ones_like(c[..., :1], dtype=c.dtype)
    matrix_b = torch.cat([c, ones], 2)  # [b, n, d + 1]

    # [b, n + d + 1, n]
    left_block = torch.cat([matrix_a, torch.permute(matrix_b, [0, 2, 1])], 1)

    num_b_cols = matrix_b.shape[2]  # d + 1
    lhs_zeros = c.new_zeros([b, num_b_cols, num_b_cols])
    right_block = torch.cat([matrix_b, lhs_zeros], 1)  # [b, n + d + 1, d + 1]
    lhs = torch.cat([left_block, right_block], 2)  # [b, n + d + 1, n + d + 1]

    rhs_zeros = c.new_zeros([b, d + 1, k])
    rhs = torch.cat([f, rhs_zeros], 1)  # [b, n + d + 1, k]

    # Then, solve the linear system and unpack the results.
    w_v = torch.linalg.solve(lhs, rhs)
    w = w_v[:, :n, :] # n
    v = w_v[:, n:, :] # d + 1

    return w, v

def _apply_interpolation(
    query_points: torch.Tensor,
    train_points: torch.Tensor,
    w: torch.Tensor,
    v: torch.Tensor,
    order: int,
) ->  torch.Tensor:
    """Apply polyharmonic interpolation model to data.
    Given coefficients w and v for the interpolation model, we evaluate
    interpolated function values at query_points.
    Args:
      query_points: `[b, m, d]` x values to evaluate the interpolation at.
      train_points: `[b, n, d]` x values that act as the interpolation centers
          (the c variables in the wikipedia article).
      w: `[b, n, k]` weights on each interpolation center.
      v: `[b, d, k]` weights on each input dimension.
      order: order of the interpolation.
    Returns:
      Polyharmonic interpolation evaluated at points defined in `query_points`.
    """

    # First, compute the contribution from the rbf term.
    pairwise_dists = _cross_squared_distance_matrix(query_points, train_points)
    phi_pairwise_dists = _phi(pairwise_dists, order)

    rbf_term = phi_pairwise_dists @ w

    # Then, compute the contribution from the linear term.
    # Pad query_points with ones, for the bias term in the linear model.
    query_points_pad = torch.cat(
        [query_points, torch.ones_like(query_points[..., :1])], 2
    )
    linear_term = query_points_pad @ v

    return rbf_term + linear_term


def interpolate_spline(
    train_points: torch.Tensor,
    train_values: torch.Tensor,
    query_points: torch.Tensor,
    order: int,
    regularization_weight: float = 0.0,
) -> torch.Tensor:
    r"""Interpolate signal using polyharmonic interpolation.
    The interpolant has the form
    $$f(x) = \sum_{i = 1}^n w_i \phi(||x - c_i||) + v^T x + b.$$
    This is a sum of two terms: (1) a weighted sum of radial basis function
    (RBF) terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term
    with a bias. The \\(c_i\\) vectors are 'training' points.
    In the code, b is absorbed into v
    by appending 1 as a final dimension to x. The coefficients w and v are
    estimated such that the interpolant exactly fits the value of the function
    at the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\),
    and the vector w sums to 0. With these constraints, the coefficients
    can be obtained by solving a linear system.
    \\(\phi\\) is an RBF, parametrized by an interpolation
    order. Using order=2 produces the well-known thin-plate spline.
    We also provide the option to perform regularized interpolation. Here, the
    interpolant is selected to trade off between the squared loss on the
    training data and a certain measure of its curvature
    ([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
    Using a regularization weight greater than zero has the effect that the
    interpolant will no longer exactly fit the training data. However, it may
    be less vulnerable to overfitting, particularly for high-order
    interpolation.
    Note the interpolation procedure is differentiable with respect to all
    inputs besides the order parameter.
    We support dynamically-shaped inputs, where batch_size, n, and m are None
    at graph construction time. However, d and k must be known.
    Args:
      train_points: `[batch_size, n, d]` float `Tensor` of n d-dimensional
        locations. These do not need to be regularly-spaced.
      train_values: `[batch_size, n, k]` float `Tensor` of n c-dimensional
        values evaluated at train_points.
      query_points: `[batch_size, m, d]` `Tensor` of m d-dimensional locations
        where we will output the interpolant's values.
      order: order of the interpolation. Common values are 1 for
        \\(\phi(r) = r\\), 2 for \\(\phi(r) = r^2 * log(r)\\)
        (thin-plate spline), or 3 for \\(\phi(r) = r^3\\).
      regularization_weight: weight placed on the regularization term.
        This will depend substantially on the problem, and it should always be
        tuned. For many problems, it is reasonable to use no regularization.
        If using a non-zero value, we recommend a small value like 0.001.
      name: name prefix for ops created by this function
    Returns:
      `[b, m, k]` float `Tensor` of query values. We use train_points and
      train_values to perform polyharmonic interpolation. The query values are
      the values of the interpolant evaluated at the locations specified in
      query_points.
    """

    # First, fit the spline to the observed data.
    w, v = _solve_interpolation(
      train_points, train_values, order, regularization_weight
    )

    # Then, evaluate the spline at the query locations.
    query_values = _apply_interpolation(query_points, train_points, w, v, order)

    return query_values


def mesh_to_image(v:torch.Tensor, uv:torch.Tensor, v2u:list, w:int ,h:int):
    """
    v: (b, n, 3) vertex coordinates
    uv: (t, 2) tex coordinates
    v2u: list or np.ndarray, vertex to tex coordinate index
    w: int, image width
    h: int, image height
    """
    v = v[:, v2u]
    uv_min = [0,0]
    uv_max = [1,1]
    x = torch.linspace(uv_min[0],uv_max[0], w).to(v.device)
    y = torch.linspace(uv_min[1],uv_max[1], h).to(v.device)
    x,y = torch.meshgrid(x,y, indexing='xy') # make meshgrid
    coord = torch.stack([x, y],-1).type(v.dtype) # stack
    coord = coord.reshape((-1,2)) # h*w, 2
    batch = v.shape[0]
    dim = v.shape[-1]
    uv = uv[None].expand((batch,-1, -1))
    coord = coord[None].expand((batch, -1, -1))
    img = interpolate_spline(uv, v, coord, 1)
    img = torch.reshape(img, (batch, h, w, dim))
    return img

def image_to_mesh(img:torch.Tensor, uv:torch.Tensor, u2v: dict, vtx_num: int):
    """
    img: (b, h, w, 3) image
    uv: (t, 2) tex coordinates
    u2v: dict, tex to vertex coordinate index
    vtx_num: int, total vertex number
    """
    b, h, w, _ = img.shape
    scale_factor = 2 # upscale image to 2x

    uv = uv * uv.new([scale_factor*w-1, scale_factor*h-1])
    uv = torch.round(uv).type(torch.long) # make uv to image array index

    # coord = np.stack([x, y],-1).astype('float32')
    # coord = coord.reshape((-1,2))
    # coord = np.repeat(coord[np.newaxis], b, 0)
    # coord = torch.Tensor(coord) # b, hw, 2
    # uvpos = interpolate_spline(coord, img, uv, 1)

    # fast array select
    img_2x = torch.nn.functional.interpolate(img.permute([0, 3, 1, 2]), scale_factor=scale_factor, mode='bilinear')
    img_2x = img_2x.permute([0, 2, 3, 1])
    uv = uv[:, 0] + uv[:, 1]*(scale_factor*w)
    uvpos = img_2x.view(b, scale_factor*scale_factor*h*w, 3)[:, uv]
    uvpos = uvpos.reshape(b, -1, 3)

    # uvpos = []
    # for x, y in uv:
        # uvpos.append(img_2x[:, y, x])
    # uvpos = torch.stack(uvpos, dim=1)
    # print(uvpos.shape)

    v = img.new_zeros([b, vtx_num, 3])
    for k, vs in u2v.items():
        v[:, k] += uvpos[:, vs].sum(dim=1) / len(vs)

    return v

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import obj_io
    print(_phi(_pairwise_squared_distance_matrix(torch.eye(3)[None]), 2))
    v, vt, f = obj_io.read_mesh("vt_template.obj")
    v_b, _, _ = obj_io.read_mesh("bareteeth.000076.obj")

    vd = v_b - v

    # print(v.shape)
    # print(vt.shape)
    # print(f.shape)
    # print(np.max(f[:, :, 0]))
    # print(np.min(f[:, :, 0]))
    # v2u = [0] * vt.shape[0]
    # for face in f:
    #     for vtxs in face:
    #         vtx, tex = vtxs
    #         v2u[tex] = vtx
    # print(v2u)
    
    # tex_nonzeros = np.bitwise_or(np.bitwise_or(vt[:, 0] == 0, vt[:, 0] == 1), np.bitwise_or(vt[:, 1] == 0, vt[:, 1] == 1))
    # tex_nonzeros = tex_nonzeros == False
    # print(tex_nonzeros)
    # print(tex_nonzeros.shape)
    # v2u = np.array(v2u)[tex_nonzeros]

    tex_nonzeros, v2u, u2v = obj_io.get_valid_tex_coords(vt, f)

    img = mesh_to_image(torch.Tensor(vd[None]), torch.Tensor(vt[tex_nonzeros]), v2u, 64, 64)
    i2v = image_to_mesh(img, torch.Tensor(vt[tex_nonzeros]), u2v, v.shape[0])
    i2v = i2v[0].numpy()
    vc = v.copy()
    vc[list(u2v.keys())] += i2v[list(u2v.keys())]

    obj_io.write_mesh('img_recon.obj', vc, f)

    print(np.linalg.norm(vd[list(u2v.keys())]-i2v[list(u2v.keys())]))
    print(i2v)
    plt.pcolormesh(img[0, ..., 2])
    plt.show()

    pts = torch.Tensor([[0, 0], [-0.5, 1], [0.5, 1], [1, 1]])
    vts = torch.Tensor([[0, 0, 0],  [-1, 0.75, 1], [0, 0.75, 1], [1, 2, 1]])
    # qs = torch.Tensor([[0, 0], [0.25, 0.25], [0.5, 0.5], [1.0, 1.0]])
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    qs = torch.Tensor(np.concatenate(np.stack([x, y], axis=-1), axis=-2))
    with torch.no_grad():
        ret = interpolate_spline(pts[None], vts[None], qs[None], 2)
    ret = ret.numpy()
    for i in range(1, 4):
        ax = plt.subplot(1, 3, i)
        ax.pcolormesh(x, y, ret[0, ..., i-1].reshape(10, 10))
    plt.show()
