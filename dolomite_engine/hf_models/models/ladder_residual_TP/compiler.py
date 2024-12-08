from functorch.compile import aot_module_simplified, make_boxed_func


def aot_backend(gm, sample_inputs):
    # Forward compiler capture
    def fw(gm, sample_inputs):
        gm.print_readable()
        with open("forward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)

    # Backward compiler capture
    def bw(gm, sample_inputs):
        gm.print_readable()
        with open("backward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)

    # Call AOTAutograd
    gm_forward = aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)

    return gm_forward
