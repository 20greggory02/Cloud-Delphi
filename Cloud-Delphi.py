import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, pi, log

# Cloud Delphi Function
def CloudDelphi(CurrentIntervals, ForegoingIntervals, ForegoingEntropy,
                Min=10, Max=80, K=6, Upper=True, Lower=True,
                Expectation=True, Uncertainty=True,
                DropExpertType=1, DropSyntecticType=3,
                DropExpertSize=0.6, DropSyntecticSize=1,
                CloudExpertIteration=2000, CloudSyntecticIteration=2000,
                CloudSyntecticColor="black", CloudExpertColor="gray",
                DecimalPlaces=2):
# Debugging Data
    IntervalError=False
    if len(CurrentIntervals) == 0 or len(CurrentIntervals) / 2 != round(len(CurrentIntervals) / 2):
        IntervalError=True
    else:
        Experts=len(CurrentIntervals) // 2

    for i in range(Experts):
        if CurrentIntervals[2*i+1] - CurrentIntervals[2*i] < 0:
            IntervalError=True

    if len(ForegoingIntervals) != 0 and len(ForegoingIntervals) != len(CurrentIntervals):
        IntervalError=True

    if len(ForegoingIntervals) == len(CurrentIntervals):
        for i in range(Experts):
            if ForegoingIntervals[2*i+1] - ForegoingIntervals[2*i] < 0:
                IntervalError=True

    if len(CurrentIntervals) != 2 and len(ForegoingEntropy) != 0 and len(ForegoingEntropy) not in [Experts, Experts+1]:
        IntervalError=True

    if len(CurrentIntervals) == 2 and len(ForegoingEntropy) > 1:
        IntervalError=True

    if IntervalError:
        raise ValueError("There is/are some error in data")

# Calculating expectation, entropy, hyper-entropy and weight
    if IntervalError:
        raise ValueError("There is/are some error in data")
    else:
        Ex_i=np.zeros(Experts)
        En_i=np.zeros(Experts)
        He_i=np.zeros(Experts)

        for i in range(Experts):
            Ex_i[i]=(CurrentIntervals[2*i+1]+CurrentIntervals[2*i])/2
            En_i[i]=(CurrentIntervals[2*i+1]-CurrentIntervals[2*i])/6

        if len(ForegoingIntervals) == 0:
            for i in range(Experts):
                He_i[i]=En_i[i]/K
        else:
            for i in range(Experts):
                He_i[i]=(max(CurrentIntervals[2*i+1]-ForegoingIntervals[2*i+1], 0) + max(ForegoingIntervals[2*i]-CurrentIntervals[2*i], 0))/6

    Ex_s=np.mean(Ex_i)
    En_s=(np.max(Ex_i+3*En_i)-np.min(Ex_i-3*En_i))/6
    He_s=np.mean(He_i)

    Weight=np.zeros(Experts)
    for i in range(Experts):
        Weight[i] = 1/(abs((Ex_i[i]-Ex_s)/Ex_s)+En_i[i]+He_i[i])
    Weight = Weight / np.sum(Weight)

    DeltaEntropyIndex = np.zeros(Experts+1)
    UncertaintyIndex = np.zeros(Experts+1)

    if len(ForegoingEntropy) != 0:
        for i in range(Experts):
            DeltaEntropyIndex[i] = abs(ForegoingEntropy[i]-En_i[i]) / ForegoingEntropy[i]
        DeltaEntropyIndex[Experts] = abs(ForegoingEntropy[Experts]-En_s) / ForegoingEntropy[Experts]
    else:
        DeltaEntropyIndex[:Experts] = En_i
        DeltaEntropyIndex[Experts] = En_s

    UncertaintyIndex[:Experts] = He_i / En_i
    UncertaintyIndex[Experts] = He_s / En_s

# Assembling data as matrix

    Round = np.column_stack((Ex_i, En_i, He_i, Weight, DeltaEntropyIndex[:Experts], UncertaintyIndex[:Experts]))
    Round = np.vstack((Round, np.array([Ex_s, En_s, He_s, 1, DeltaEntropyIndex[Experts], UncertaintyIndex[Experts]])))

    columns = ["Ex", "En", "He", "w", "ΔEn", "Unc"]
    rows = [f"Exp_{i+1}" for i in range(Experts)] + ["Exp_s"]

    Round = pd.DataFrame(Round, index=rows, columns=columns).T

# Preparing coordinate system
    plt.figure(figsize=(16, 9))
    plt.xlim(Min, Max)
    plt.ylim(0, 1.03)
    plt.xlabel("x")
    plt.ylabel("μ(x)")
    fig = plt.gcf()
    exp_label = "expert" if Experts == 1 else "experts"
    fig.suptitle(f"Cloud Delphi - {Experts} {exp_label}", fontsize=16, y=1)

# Generating the Cloud function

    def Cloud(Ex, En, He, Color, Type, Size, N, Min, Max,
              CurveE=False, CurveU=False, CurveL=False, CurveI=False):
        P = np.zeros((N, 2))
        for i in range(N):
            Em_i = abs(np.random.normal(En, He))  # abs to avoid negative deviation
            X_i = np.random.normal(Ex, Em_i)
            Mu_i = sqrt(2*pi)*Em_i*(1/(Em_i*sqrt(2*pi)))*np.exp(-0.5*((X_i - Ex)/Em_i)**2)
            P[i, 0], P[i, 1] = X_i, Mu_i

        plt.scatter(P[:, 0], P[:, 1], color=Color, s=Size*10)

        D1 = En-3*He
        D2 = En+3*He

        x_vals = np.linspace(Min, Max, 400)
        E = lambda x: sqrt(2*pi)*En*(1/(En*sqrt(2*pi)))*np.exp(-0.5*((x-Ex)/En)**2)
        U = lambda x: sqrt(2*pi)*D2*(1/(D2*sqrt(2*pi)))*np.exp(-0.5*((x-Ex)/D2)**2)
        L = lambda x: sqrt(2*pi)*D1*(1/(D1*sqrt(2*pi)))*np.exp(-0.5*((x-Ex)/D1)**2)
        I = lambda x: U(x)-L(x)

        if CurveE: plt.plot(x_vals, E(x_vals), color="green")
        if CurveU: plt.plot(x_vals, U(x_vals), color="red", linestyle="--", linewidth=1)
        if CurveL: plt.plot(x_vals, L(x_vals), color="blue", linestyle="--", linewidth=1)
        if CurveI:
            plt.plot(x_vals, I(x_vals), color="magenta")
            D3 = 2*D2*D1*sqrt((log(D2)-log(D1))/(D2**2-D1**2))
            X1, X2 = Ex - D3, Ex + D3
            plt.vlines([X1, X2], L(X1), U(X1), color="magenta", linewidth=3)
            plt.text(X1 + (Max - Min) * 0.13, (L(X1) + U(X1)) / 2, "Maximum uncertainty", color="magenta")
            plt.arrow(X1 + (Max - Min) * 0.005, (L(X1) + U(X1)) / 2, (Max - Min) * 0.05, 0, color="magenta", head_width=0.03)

        if any([CurveE, CurveU, CurveL, CurveI]):
            labels = [lbl for lbl, flag in zip(["Expectation", "Upper", "Lower", "Uncertainty"], [CurveE, CurveU, CurveL, CurveI]) if flag]
            colors = [c for c, flag in zip(["green", "red", "blue", "magenta"], [CurveE, CurveU, CurveL, CurveI]) if flag]
            plt.legend(labels, loc="best", frameon=True, fontsize=8, facecolor="white", edgecolor="black", labelcolor=colors)

# Plotting each expert's clouds and the synthetic cloud

    if Experts != 1:
        for i in range(Experts):
            Cloud(
                Round.loc["Ex", f"Exp_{i+1}"],
                Round.loc["En", f"Exp_{i+1}"],
                Round.loc["He", f"Exp_{i+1}"],
                CloudExpertColor,
                DropExpertType,
                DropExpertSize,
                CloudExpertIteration,
                Min,
                Max
            )

    Cloud(
        Round.loc["Ex", "Exp_s"],
        Round.loc["En", "Exp_s"],
        Round.loc["He", "Exp_s"],
        CloudSyntecticColor,
        DropSyntecticType,
        DropSyntecticSize,
        CloudSyntecticIteration,
        Min,
        Max,
        CurveU=Upper,
        CurveL=Lower,
        CurveE=Expectation,
        CurveI=Uncertainty
    )

    if Experts != 1:
        plt.annotate(
            "Exp_s",
            xy=(Round.loc["Ex", "Exp_s"], 1.0),
            xytext=(0, 25),
            textcoords='offset points',
            ha='center',
            color=CloudSyntecticColor,
            arrowprops=dict(arrowstyle='-', lw=0.7, color=CloudSyntecticColor)
        )

        for i in range(Experts):

            offset_y = 50 if (i % 2 == 0) else 30 # Vertical adjustment for each expertin the final plot (Prevent label overlay)

            plt.annotate(
                f"Exp_{i+1}",
                xy=(Round.loc["Ex", f"Exp_{i+1}"], 1.0),
                xytext=(0, offset_y),
                textcoords='offset points',
                ha='center',
                color=CloudExpertColor,
                arrowprops=dict(arrowstyle='-', lw=0.7, color=CloudExpertColor)
            )
    else:
        plt.annotate(
            "Normal cloud generated",
            xy=(Round.loc["Ex", "Exp_s"], 1.0),
            xytext=(0, 25),
            textcoords='offset points',
            ha='center',
            color=CloudSyntecticColor
        )

    # Constructing the output matrix

    return Round.round(DecimalPlaces)

# An example of a synthetic cloud based on six experts

a = np.array([58,64,  20,26,  20,30,  31,33,  62,68,  71,79])  # CurrentIntervals
b = np.array([57,63,  19,23,  20,31,  32,33,  61,72,  73,82])  # ForegoingIntervals
c = np.array([0.61, 0.30, 1.00, 0.22, 1.09, 1.50, 1.95])       # ForegoingEntropy

Round = CloudDelphi(a, b, c)
print(Round)
plt.show()