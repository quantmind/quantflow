import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from app.utils import nav_menu
    nav_menu()
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Deep Implied Volatility Factor Model
    """)
    return


@app.cell
def _():
    import numpy as np
    import torch

    from quantflow.options.divfm.network import DIVFMNetwork
    from quantflow.options.divfm.trainer import DayData, DIVFMTrainer
    from quantflow.options.pricer import OptionPricer
    from quantflow.sp.heston import HestonJ
    from quantflow.utils.distributions import DoubleExponential

    # ---------------------------------------------------------------------------
    # Grid settings
    # ---------------------------------------------------------------------------

    TTM_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]
    MAX_MONEYNESS_TTM = 1.5  # moneyness_ttm range for sampling and pricing
    N_PER_TTM = 20  # random options sampled per TTM per day

    # ---------------------------------------------------------------------------
    # HestonJ parameter ranges (uniform sampling)
    # ---------------------------------------------------------------------------

    PARAM_RANGES = {
        "vol": (0.10, 0.70),
        "rho": (-0.80, 0.10),
        "kappa": (0.50, 5.00),
        "sigma": (0.20, 1.50),
        "jump_fraction": (0.1, 0.50),
        "jump_asymmetry": (-0.50, 0.50),
    }


    # ---------------------------------------------------------------------------
    # Fixture generation
    # ---------------------------------------------------------------------------


    def _make_pricer(rng: np.random.Generator) -> OptionPricer:
        """Sample a random HestonJ parameter set and return a ready pricer."""
        vol = float(rng.uniform(*PARAM_RANGES["vol"]))
        rho = float(rng.uniform(*PARAM_RANGES["rho"]))
        kappa = float(rng.uniform(*PARAM_RANGES["kappa"]))
        sigma = float(rng.uniform(*PARAM_RANGES["sigma"]))
        jump_fraction = float(rng.uniform(*PARAM_RANGES["jump_fraction"]))
        jump_asymmetry = float(rng.uniform(*PARAM_RANGES["jump_asymmetry"]))
        sv = sigma/vol
        kappa = max(kappa, 0.6*sv*sv)

        model = HestonJ.create(
            DoubleExponential,
            vol=vol,
            kappa=kappa,
            rho=rho,
            sigma=sigma,
            jump_fraction=jump_fraction,
            jump_asymmetry=jump_asymmetry,
        )
        return OptionPricer(model=model, max_moneyness_ttm=MAX_MONEYNESS_TTM)


    def _sample_day(rng: np.random.Generator, pricer: OptionPricer) -> DayData | None:
        """Price options at random (moneyness_ttm, ttm) points and return DayData.

        Returns None if all points are invalid (e.g. numerical pricing failure).
        """
        m_list: list[np.ndarray] = []
        t_list: list[np.ndarray] = []
        iv_list: list[np.ndarray] = []

        for ttm in TTM_GRID:
            mat = pricer.maturity(ttm)
            m_ttm = rng.uniform(-MAX_MONEYNESS_TTM, MAX_MONEYNESS_TTM, N_PER_TTM).astype(
                np.float32
            )
            moneyness = m_ttm * np.sqrt(ttm)
            ivs = np.interp(moneyness, mat.moneyness, mat.implied_vols)

            # drop any degenerate points (NaN / non-positive IV)
            valid = np.isfinite(ivs) & (ivs > 0)
            if not valid.any():
                continue

            m_list.append(m_ttm[valid])
            t_list.append(np.full(valid.sum(), ttm, dtype=np.float32))
            iv_list.append(ivs[valid].astype(np.float64))

        if not m_list:
            return None

        return DayData(
            moneyness_ttm=np.concatenate(m_list),
            ttm=np.concatenate(t_list),
            implied_vols=np.concatenate(iv_list),
        )


    def generate_fixtures(
        num_days: int = 300,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[DayData]:
        """Generate *num_days* synthetic IV days from random HestonJ parameters.

        Each day is a different random parameter set, giving the DIVFM model a
        diverse training distribution that covers varying vol levels, skews, and
        term structures.
        """
        rng = np.random.default_rng(seed)
        days: list[DayData] = []
        skipped = 0

        for i in range(num_days):
            pricer = _make_pricer(rng)
            day = _sample_day(rng, pricer)
            if day is None:
                skipped += 1
            else:
                days.append(day)

            if verbose and (i + 1) % 50 == 0:
                print(f"  generated {i + 1}/{num_days} parameter sets  ({len(days)} valid)")

        if verbose:
            print(f"Fixture generation done: {len(days)} valid days, {skipped} skipped")

        return days


    def fit_divfm(
        days: list[DayData],
        num_factors: int = 5,
        hidden_size: int = 32,
        num_hidden_layers: int = 3,
        lr: float = 1e-3,
        batch_days: int = 32,
        num_steps: int = 500,
        val_fraction: float = 0.1,
        seed: int = 0,
        log_every: int = 50,
    ) -> tuple[DIVFMNetwork, list[float]]:
        """Train a DIVFMNetwork on the given days.

        Splits days into train/val, trains the network, and returns the trained
        network together with the per-step training losses.
        """
        torch.manual_seed(seed)

        n_val = max(1, int(len(days) * val_fraction))
        train_days = days[n_val:]
        val_days = days[:n_val]

        net = DIVFMNetwork(
            num_factors=num_factors,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
        )
        trainer = DIVFMTrainer(net, lr=lr, batch_days=batch_days)

        print(
            f"Training DIVFM  factors={num_factors}  hidden={hidden_size}"
            f"  layers={num_hidden_layers}  lr={lr}"
            f"  batch_days={batch_days}  steps={num_steps}"
        )
        print(f"  train days: {len(train_days)}  val days: {len(val_days)}")

        losses = trainer.fit(
            train_days,
            num_steps=num_steps,
            val_days=val_days,
            log_every=log_every,
        )

        val_loss = trainer.evaluate(val_days)
        print(f"Final val loss: {val_loss:.6f}")

        return net, losses


    return fit_divfm, generate_fixtures, np, torch


@app.cell
def _(generate_fixtures):
    days = generate_fixtures(num_days=300, seed=42)
    return (days,)


@app.cell
def _(days, fit_divfm):
    net, losses = fit_divfm(days, num_steps=500, log_every=50)
    return (net,)


@app.cell
def _():
    return


@app.cell
def _(mo, net, np, torch):
    import plotly.graph_objects as go

    # 1. Create the coordinate grid
    m_range = np.linspace(-1.5, 1.5, 40)  # moneyness_ttm
    t_range = np.linspace(0.1, 2.0, 40)   # ttm
    M, T = np.meshgrid(m_range, t_range)

    # Flatten the grid to feed into the neural network
    M_flat = M.flatten()
    T_flat = T.flatten()

    # Prepare inputs for the network
    M_tensor = torch.tensor(M_flat, dtype=torch.float32)
    T_tensor = torch.tensor(T_flat, dtype=torch.float32)

    # 2. Evaluate the network to get the factors
    with torch.no_grad():
        factors_pred = net(M_tensor, T_tensor).numpy()

    # 3. Create a Plotly figure for factors 1, 2, 3, and 4
    tabs_dict = {}
    for i in range(1, 5):
        # Reshape the 1D factor output back into the 2D grid shape
        Z = factors_pred[:, i].reshape(M.shape)
    
        fig = go.Figure(data=[go.Surface(x=M, y=T, z=Z, colorscale='Viridis')])
    
        fig.update_layout(
            title=f"DIVFM Learned Factor {i}",
            scene=dict(
                xaxis_title='Moneyness / √TTM',
                yaxis_title='Time to Maturity',
                zaxis_title=f'Factor {i} Value',
                camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)),
                dragmode="turntable"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    
        tabs_dict[f"Factor {i}"] = fig

    # 4. Display them in an interactive tabbed interface
    mo.ui.tabs(tabs_dict)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
