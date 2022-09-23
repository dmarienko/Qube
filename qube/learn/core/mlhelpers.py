import dill
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from qube.learn.core.base import MarketDataComposer
from qube.learn.core.pickers import SingleInstrumentPicker
from qube.learn.core.utils import ls_params
from qube.simulator.utils import generate_simulation_identificator
from qube.utils.nb_functions import z_save
from qube.utils.utils import mstruct


def gridsearch(experiment_id, estimator, scoring, data, param_grid={},
               t_range=None,
               on='close', ts_splits=3, verbose=True,
               return_train_score=True,
               save=False):
    """
    Run sklearn's gridsearch CV using TimeSeriesSplit splitter

    Example
    -------

    m = run_grid('Experiment1',
             -FisherRfiEntries(8, 4, 16, '1h'),
             ReverseSignalsSharpeScoring(commissions='binance'),
             ohlc,
             param_grid = qube.simulator.utils.permutate_params({
                 'predictor__timeframe': ['1H'],
                 'predictor__period': [4,5,6,7,8,9],
                 'predictor__lower': [3,4,5,6,7,8,9,10,11,12],
                 'predictor__upper': [17,16,15,14,13,12]
                 }, conditions=lambda  predictor__lower,predictor__upper: predictor__lower+predictor__upper==20),
             t_range=None, ts_splits=5, verbose=True, return_train_score=True, save=False)

    signals = m.predict(ohlc)
    returns = m.estimated_portfolio(ohlc, ReverseSignalsSharpeScoring(commissions='binance'))
    plt.plot(returns.sum(axis=1).cumsum())

    
    """
    if verbose:
        print('- Default parameters -')
        ls_params(estimator)
        print('\n')
        print(f'> Started {experiment_id} / {ts_splits} splits on {on}')

    gs = GridSearchCV(n_jobs=1, cv=TimeSeriesSplit(ts_splits), estimator=estimator, scoring=scoring,
                      param_grid=param_grid, verbose=verbose, return_train_score=return_train_score)

    picker = SingleInstrumentPicker()
    if t_range is not None and len(t_range) == 2:
        picker.for_range(*t_range)

    mds = MarketDataComposer(gs, picker, column=on)
    mds.fit(data, None)
    if verbose:
        print(f'> best scored ({gs.best_score_:.3f}) parameteres: ')
        print(gs.best_params_)

    if save:
        sid = generate_simulation_identificator(experiment_id, 'gridsearch', pd.Timestamp('now'))
        pth = f'{experiment_id}/gridsearch/{sid}'
        z_save(pth, mstruct(eid=experiment_id,
                            splits=ts_splits,
                            ranges=t_range,
                            on=on,
                            params=gs.best_params_,
                            score=gs.best_score_,
                            scoring=dill.dumps(scoring),
                            estimator=dill.dumps(gs.best_estimator_)))
    return mds
