using Plots, TaylorSeries


function unrolled_heat_with_scaling(v, m, b, x, y, range, center; s=.01)
    """The taylor series library seems to not be able to handle
    the functions returned by the scale_function method. This combines the
    math done by that method with the heat calculation. """
    sqrt(
        (
            1/(s +
                sqrt(((v*(range/2) + center) - x)^2 + ((m*(v*(range/2) + center) + b) - y)^2)
            ^2)
        )^2 +
        (
            m/(s +
                sqrt(((v*(range/2) + center) - x)^2 + ((m*(v*(range/2) + center) + b) - y)^2)
            ^2)
        )^2
    )
end

function alt1(v, m, b, x, y, range, center; s=.01)
    """The taylor series library seems to not be able to handle
    the functions returned by the scale_function method. This combines the
    math done by that method with the heat calculation. """

            1/(1 + s +
            sqrt(
                ((v*(range/2) + center) - x)^2 + ((m*(v*(range/2) + center) + b) - y)^2
                )
                ^2)
end

function graph_appr(f; m=1.0, b=0.0, x=1.0, y=2.0, r=20.0, c=1.0, order=30)
    t = Taylor1(typeof(r), order)
    ts_polynomial = f(t, m, b, x, y, r, c)
    ts(v) = evaluate(ts_polynomial, v)
    truth(v) = f(v, m, b, x, y, r, c)
    plot(truth, -.8, .8)
    # plot!(ts, -.3, .3)
end

function hmm(t, m, b, x, y, r, c)
    1/log(
    2 + (t*10+.2)^2
    )
end
graph_appr(hmm)
