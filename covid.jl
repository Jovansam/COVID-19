
using JuMP, Plots
using Ipopt
using KNITRO
using ReSHOP

function covid_19()


    A = 1.0
    Λ = 0.0

    β = 1.0    # sets the speed with which people get infected. The higher the more people are infected

    γ₁ = 0.20   # fraction of infected who recover in period
    γ₂ = 1.00   # fraction of the that builds long-term immunity

    μ = 0.30    # death rate from infection
    ρ = 0.01    # discount rate

    χ = 0.3     # elasticity of marginal utility; higher means full employment is better; in this case the government is more willing to shutdown employment in some periods to ensure overall fully employment; the lowe tis figure is the less likely or prolonged the goverment shuts down

    ζ₁ = 0.00  # sets the disutility from death
    ζ₂ = 2

    α₁ = 0
    α₂ = 1.0
    α₃ = 0

    T = 100;

    #covid = Model(Ipopt.Optimizer)
    covid = Model(KNITRO.Optimizer)
    #covid = direct_model(ReSHOP.Optimizer(solver="knitro"))
    #covid = Model(optimizer_with_attributes(ReSHOP.Optimizer, "solver"=>"conopt3"))

    @variable(covid, 0<=e[t=1:T]<=1)
    @variable(covid, S[t=1:(T+1)]>=0)
    @variable(covid, I[t=1:(T+1)]>=0)
    @variable(covid, R[t=1:(T+1)]>=0)
    @variable(covid, N[t=1:T]>=0)

    @NLconstraint(covid, [t=1:T], S[t+1] - S[t] == Λ - μ*α₁*S[t] - β*(I[t]/N[t])*S[t]*e[t] + (1-γ₂)*R[t])
    @NLconstraint(covid, [t=1:T], I[t+1] - I[t] == β*(I[t]/N[t])*S[t]*e[t] - γ₁*I[t] - μ*α₂*I[t])
    @constraint(covid, [t=1:T], R[t+1] - R[t] == γ₁*I[t] - μ*α₃*R[t] - (1-γ₂)*R[t])

    @constraint(covid, [t=1:T], N[t] == I[t] + R[t] + S[t])

    @constraint(covid, [t=1], S[t] == 0.99)
    @constraint(covid, [t=1], I[t] == 0.01)
    @constraint(covid, [t=1], R[t] == 0.0)

    @NLobjective(covid, Max, sum( ((1+ρ)^(-t)) * (A*e[t]^χ*(S[t] + R[t])) - ζ₁*(N[1] - N[t])^ζ₂  for t in 1:T ) )

    JuMP.optimize!(covid)
    e,S,I,R,N = JuMP.value.(e),JuMP.value.(S),JuMP.value.(I),JuMP.value.(R),JuMP.value.(N)
    S = S[1:T]
    R = R[1:T]
    I = I[1:T]
    return e,S,I,R,N,T
end


function plot_figures()
    e,S,I,R,N,T = covid_19()

    @show e
    plt1 = plot(1:Int(0.1*T), e[1:Int(0.1*T)],label="e",ylim=(0,1),markershape=:h)
    plt2 = plot(1:T, S./N,label="S/N",ylim=(0,1))
    plt3 = plot(1:T, I./N,label="I/N",ylim=(0,1))
    plt4 = plot(1:T, R./N,label="R/N",ylim=(0,1))
    plt5 = plot(1:T, (S+R)./N,label="(S+R)/N",ylim=(0,1))
    plt6 = plot(1:T, N,label="N",ylim=(0,1))
    plot(plt1,plt2,plt3,plt4,plt5,plt6)
end

plot_figures()
