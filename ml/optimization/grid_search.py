import numpy as np



def find_max(f, ranges, step_sizes, print_progress = True, print_progress_freq = 1):
    axes_steps_shapes = __get_axes_steps_shapes(ranges, step_sizes)
    #if the search grid is big enough, max_n will be too big to fit in an integer.
    #find_max attempts to split the region in to pieces first, then calls this method
    #for a final answer once this overflow wouldn't occur anymore, then compares the
    #solutions for the split up regions
    max_n = np.prod(axes_steps_shapes)
    if max_n == 0:
        raise ValueError("More grid inputs than can fit in a 64 bit int. Overflow occurred! (This grid search would take a super long time anyway)")
    if print_progress:
        print("number of grid search iterations required: ", max_n)
    best_f = None
    best_x = None
    for n in range(0, max_n):
        x = __convert_step_num_to_x(n, axes_steps_shapes, ranges, step_sizes)
        assert((x <= ranges[:,1]).all()), x
        f_x = f(x)
        if best_f is None or f_x > best_f:
            best_f = f_x
            best_x = x
        if n % print_progress_freq == 0 and print_progress:
            print("percent completion: ", 100 * (n / max_n))
            print("current input: ", x)
            print("current cost: ", f_x)
            print("best cost: ", best_f)
            print("best input: ", best_x)
            print("--------------------------------------")
    return best_x


def __get_axes_steps_shapes(ranges, step_sizes):
    out = np.zeros(step_sizes.shape[0], dtype = np.int)
    x = ranges[:,0].copy()
    while (x <= ranges[:,1]).any():
        out[np.where(x <= ranges[:,1])] += 1
        x += step_sizes
    return out
    '''out = ((ranges[:,1] - ranges[:,0]) / step_sizes).astype(np.int) + 3
    where_can_go_out_of_bounds = np.where(ranges[:,0] + step_sizes * out > ranges[:,1])
    out[where_can_go_out_of_bounds] -= 1
    return out'''

def __convert_step_num_to_x(n, axes_steps_shapes, ranges, step_sizes):
    axes_steps = __convert_step_num_to_steps_along_axes(n, axes_steps_shapes)
    return ranges[:,0] + step_sizes * axes_steps

def __convert_step_num_to_steps_along_axes(n, axes_steps_shape):
    return np.unravel_index(n, axes_steps_shape)


if __name__ == "__main__":
    def f(x):
        return -np.sum(np.square(x))

    x_size = 5
    ranges = np.zeros((x_size,2), dtype = np.int64)
    ranges[:,0] = -10
    ranges[:,1] = 10
    step_sizes = np.ones(x_size)

    print("optimal: ", find_max(f, ranges, step_sizes))
