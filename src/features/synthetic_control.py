import datetime

from loguru import logger

import torch
import numpy as np
import pandas as pd

from src.models.pytorch_synthetic_control import MyModel
from src.params import Config
from src.utils import read_config, log_block, timeit
from typing import List, Optional, Tuple, Dict
import scipy.stats as sts


def split_by_date(
        synth_df: pd.DataFrame, on_date: datetime.date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбить датафрейм по дате on_date.

    :param synth_df: датафрейм с datetime индексом
    :param on_date: по какой дате бить
    :return:
    """
    return (
        synth_df[synth_df.index.date < on_date],
        synth_df[synth_df.index.date >= on_date],
    )


def sensitivity_check(
        X_post: np.ndarray,
        y_post: np.ndarray,
        coeffs: np.ndarray,
        label: Optional[str] = None,
        boundary: Optional[float] = 1e4,
):
    config = read_config()
    results = []
    np.random.seed(config.general.random_state)
    for mean in np.linspace(-boundary, boundary, 1000):
        if mean == 0.0:
            continue
        delta = np.random.normal(loc=mean, scale=X_post.mean().mean() * 0.1)
        results.append(
            (label, mean, sts.ttest_ind(np.dot(X_post, coeffs), y_post + delta)[1])
        )
    results.append((label, 0.0, sts.ttest_ind(np.dot(X_post, coeffs), y_post)[1]))
    return results


def split_by_group(
        synth_df: pd.DataFrame, test_group: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбить датафрейм по группе клиентов.

    :param synth_df: кого передать в тестовую группу
    :param test_group: список колонок, которые отправить в тестовую группу
    :return: (контроль, тест)
    """
    test_group = list(test_group)
    return synth_df.drop(columns=test_group), synth_df[test_group]


@timeit
def synthetic_control(
        X_pre: np.ndarray,
        y_pre: np.ndarray,
        lr: float,
        loss_func: str,
        epochs: int = 2000,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    x = torch.from_numpy(X_pre.astype(np.float32))
    y = torch.from_numpy(y_pre.astype(np.float32)).reshape(y_pre.shape[0])
    if loss_func == "mse":
        crit = torch.nn.MSELoss(reduction="sum")
    elif loss_func == "mae":
        crit = torch.nn.L1Loss(reduction="sum")
    else:
        raise NotImplementedError()
    model = MyModel(X_pre, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss = torch.randn(())
    for t in range(epochs):
        y_pred = model(x)
        loss = crit(y_pred, y)
        if t % 1000 == 999:
            logger.debug(f"iteration {t} with loss {loss.item()}")
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (
        torch.softmax(list(model.parameters())[0].T.detach(), dim=-1).numpy(),
        loss.detach().numpy(),
    )


def cuped(
        config: Config,
        X_pre: pd.DataFrame,
        X_post: pd.DataFrame,
        y_pre: pd.DataFrame,
        y_post: pd.DataFrame,
        all_pre: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выполнить CUPED над данным.

    :param config: объект :class:`Config`
    :param X_pre:
    :param X_post:
    :param y_pre:
    :param y_post:
    :param all_pre: все до даты трита (чтобы точнее подсчитать матожидание)
    :return:
    """
    if not config.control.cuped.enabled:
        logger.info("skipping CUPED as specified in config file")
        return X_pre, X_post, y_pre, y_post
    n_days = config.control.cv.n_future_days
    do_time_based = config.control.cuped.time_based
    if do_time_based:
        X_pre, y_pre = X_pre.copy().iloc[-n_days:], y_pre.copy().iloc[-n_days:]
        X_post, y_post = X_post.copy().iloc[:n_days], y_post.copy().iloc[:n_days]
    axis = 1 if do_time_based else 0
    cov_pre = X_pre.mean(axis=axis)
    theta = 1
    cov_pre_mean = all_pre.mean().mean()
    X_pre_cuped = X_pre.subtract(theta * (cov_pre - cov_pre_mean), axis=1 - axis)
    y_pre_cuped = y_pre.subtract(
        theta * (y_pre.mean(axis=axis) - y_pre.mean(axis=axis).mean(axis=0)),
        axis=1 - axis,
    )
    X_post_cuped = X_post.subtract(
        theta * (cov_pre - cov_pre_mean).values, axis=1 - axis
    )
    y_post_cuped = y_post.subtract(
        theta * (y_pre.mean(axis=axis) - y_pre.mean(axis=axis).mean(axis=0)).values,
        axis=1 - axis,
    )
    for i in (X_pre_cuped, X_post_cuped, y_pre_cuped, y_post_cuped):
        assert not i.isna().any().any()
    return X_pre_cuped, X_post_cuped, y_pre_cuped, y_post_cuped


def group_based_cv(
        synth_df: pd.DataFrame, split_date: Optional[datetime.date] = None
) -> Tuple[
    pd.DataFrame,
    Tuple[
        Dict[int, pd.DataFrame],
        Dict[int, pd.DataFrame],
        Dict[int, pd.DataFrame],
        Dict[int, pd.DataFrame],
        pd.DataFrame,
    ],
]:
    config = read_config()
    if split_date is None:
        split_date = config.control.model_start_date
    rv_x_pre, rv_x_post, rv_y_pre, rv_y_post = {}, {}, {}, {}
    synth_pre, synth_post = split_by_date(synth_df, on_date=split_date)
    X_pre, y_pre = split_by_group(synth_pre, test_group=config.control.treated_clients)
    X_post, y_post = split_by_group(
        synth_post, test_group=config.control.treated_clients
    )
    if config.control.cv.use_cuped:
        _X_pre, _X_post, _y_pre, _y_post = cuped(
            config=config,
            X_pre=X_pre,
            X_post=X_post,
            y_pre=y_pre,
            y_post=y_post,
            all_pre=synth_pre,
        )
    else:
        _X_pre, _X_post, _y_pre, _y_post = (
            X_pre.copy(),
            X_post.copy(),
            y_pre.copy(),
            y_post.copy(),
        )
    _y_pre = _y_pre.mean(axis=1)
    _y_post = _y_post.mean(axis=1)
    rv_x_pre[0] = _X_pre
    rv_x_post[0] = _X_post
    rv_y_pre[0] = _y_pre
    rv_y_post[0] = _y_post
    N_, loss = config.control.n_placebo, "mae"

    results, reg_ = [], None
    regs_array = []
    # Реальная тестовая группа
    with log_block("adding real test group"):
        reg, _ = synthetic_control(
            X_pre=_X_pre.values,
            y_pre=_y_pre.values,
            loss_func="mae",
            lr=config.control.torch.learning_rate,
            epochs=config.control.torch.epochs,
            seed=config.general.random_state,
        )
        results.extend(
            sensitivity_check(
                X_post=_X_post.values, y_post=_y_post, coeffs=reg, label="real"
            )
        )
        regs_array.append((0, list(reg)))
    GROUP_SIZE = len(config.control.treated_clients)
    for i in range(N_):
        seed = config.general.random_state + i
        group = list(X_pre.sample(GROUP_SIZE, axis=1, random_state=seed).columns)
        logger.info(f"placebo with group={group}")
        X_placebo_pre, y_placebo_pre = split_by_group(X_pre, test_group=group)
        X_placebo_pre["mean_treated"] = y_pre.mean(axis=1)
        X_placebo_post, y_placebo_post = split_by_group(X_post, test_group=group)
        X_placebo_post["mean_treated"] = y_post.mean(axis=1)
        if config.control.cv.use_cuped:
            X_placebo_pre, X_placebo_post, y_placebo_pre, y_placebo_post = cuped(
                config=config,
                X_pre=X_placebo_pre,
                X_post=X_placebo_post,
                y_pre=y_placebo_pre,
                y_post=y_placebo_post,
                all_pre=synth_pre,
            )

        y_placebo_pre = y_placebo_pre.mean(axis=1)
        y_placebo_post = y_placebo_post.mean(axis=1)
        rv_x_pre[seed] = X_placebo_pre
        rv_x_post[seed] = X_placebo_post
        rv_y_pre[seed] = y_placebo_pre
        rv_y_post[seed] = y_placebo_post
        reg_, _ = synthetic_control(
            X_placebo_pre.values,
            y_placebo_pre.values,
            lr=config.control.torch.learning_rate,
            epochs=config.control.torch.epochs,
            seed=config.general.random_state,
            loss_func=loss,
        )
        results.extend(
            sensitivity_check(
                X_post=X_placebo_post.values,
                y_post=y_placebo_post.values.reshape(-1),
                coeffs=reg_,
                label=f"fake group with seed={seed}",
                boundary=10 * X_pre.mean().mean(),
            )
        )
        regs_array.append((seed, list(reg_)))
    regs_rv = pd.DataFrame(regs_array, columns=["seed", "coeffs"])
    return pd.DataFrame(results, columns=["hue", "mean", "p_value"]), (
        rv_x_pre,
        rv_x_post,
        rv_y_pre,
        rv_y_post,
        regs_rv,
    )


def rsearch_intercept(df: pd.DataFrame):
    """Ищет правую точку пересечения графика p-value от mean с желаемым alpha в
    конфиге. Это соответствует чувствительности.

    :param df: датафрейм с колонками [x_label, mean, p-value] - \
    дата сплита на cv фолде, матожидание шума и полученный p-value

    :return: тот же df, но на каждую дату добавлена колонка sens
    """
    config = read_config()
    x_label = "hue"
    tmp_ = df.sort_values([x_label, "mean"]).query("mean != -1.0")
    _results = []
    for dt in tmp_[x_label].unique():
        tmp__ = tmp_.loc[tmp_[x_label] == dt]
        for row in tmp__[::-1].itertuples():
            if row.p_value > config.control.alpha:
                break
            mm = row
        _results.append((dt, mm.mean))
    for (dt, sens) in _results:
        df.loc[df[x_label] == dt, "sens"] = sens
    return df


def run(config: Config, target: str):

    synth_df = pd.DataFrame()
    results, (X_pre_d, X_post_d, y_pre_d, y_post_d, reg) = group_based_cv(synth_df)
    rsearch_intercept(results)
    results.to_csv(f"cv_results_{target}.csv")


if __name__ == "__main__":
    conf = read_config()
    for target in ("target_1", "target_2"):
        with log_block(f"running target {target}"):
            run(config=conf, target=target)
