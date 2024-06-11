import unittest
from unittest import main
from unittest.mock import patch
import mongomock
from mongomock.gridfs import enable_gridfs_integration

enable_gridfs_integration()

from qube.tests.utils_for_tests import (
    _read_timeseries_data,
    _init_mongo_db_with_market_data,
)
from qube.datasource.controllers.MongoController import MongoController
from qube.booster.core import Booster
from qube.utils.nb_functions import z_ls, z_ld
import numpy as np
import pytest
from pytest import fixture


@fixture(autouse=True)
def mock_pool_imap_unordered(monkeypatch):
    """
    Making single process from multiproc for being able to use mocked mongo
    """

    def _mock_start(obj):
        obj._target(*obj._args, **obj._kwargs)

    def _mock_join(obj):
        pass

    def _mock_imap_unordered(obj, func, args):
        return [func(a) for a in args]

    monkeypatch.setattr(
        "multiprocess.pool.Pool.imap_unordered",
        lambda self, func, args=(): _mock_imap_unordered(self, func, args),
    )
    monkeypatch.setattr(
        "multiprocessing.pool.Pool.imap_unordered",
        lambda self, func, args=(): _mock_imap_unordered(self, func, args),
    )
    monkeypatch.setattr(
        "multiprocess.context.Process.start", lambda self: _mock_start(self)
    )
    monkeypatch.setattr(
        "multiprocess.context.Process.join", lambda self: _mock_join(self)
    )


class BoosterTest(unittest.TestCase):

    @mongomock.patch()
    def test_basic_booster(self):
        _init_mongo_db_with_market_data("md")

        boo = Booster("qube/tests/booster/booster_portfolio_task.yml", log=True)
        print(boo.get_all_entries())

        boo.task_portfolio("TestOneByOne", run=True, save_to_storage=True)
        print(z_ls("portfolios/.*", dbname="booster"))
        print(z_ls("runs/.*", dbname="booster"))
        print(z_ls("stats/.*", dbname="booster"))

        # - test report -
        rep0 = z_ld("portfolios/BooTest/TestOneByOne", dbname="booster")["report"]
        print(rep0)
        np.testing.assert_almost_equal(
            rep0["Set0"]["Gain"].values,
            [-102.43835902336497, -1340.832626, -1443.270985, -3.469401],
            decimal=6,
        )
        np.testing.assert_almost_equal(
            rep0["Set1"]["Execs"].values, [216.0, 164.0, 380.0, np.nan], decimal=1
        )

    @mongomock.patch()
    def test_portfolio_config(self):
        _init_mongo_db_with_market_data("md")
        boo = Booster("qube/tests/booster/booster_portfolio_new.yml", log=True)
        # print(boo.get_all_entries())

        boo.task_portfolio("TestPortfolio", run=True, save_to_storage=True)
        print(z_ls("portfolios/.*", dbname="booster"))
        print(z_ls("runs/.*", dbname="booster"))
        print(z_ls("stats/.*", dbname="booster"))
        # print(z_ld('runs/BooTest/sim.0.ETHUSDT/TestPortfolio_PORTFOLIO', dbname='booster').result.executions)
        # print(z_ld('runs/BooTest/sim.0.SOLUSDT/TestPortfolio_PORTFOLIO', dbname='booster').result.executions)

        print(
            z_ld(
                "runs/BooTest/sim.0.(PORTFOLIO)/TestPortfolio_PORTFOLIO",
                dbname="booster",
            ).result.executions
        )

        # - test report -
        rep0 = z_ld("portfolios/BooTest/TestPortfolio", dbname="booster")["report"]
        print(rep0["Set0"])
        print(rep0["Set1"])
        np.testing.assert_almost_equal(
            rep0["Set0"]["Gain"].values, [703.166439, 703.166439, 87.895805], decimal=6
        )
        np.testing.assert_almost_equal(
            rep0["Set1"]["Gain"].values,
            [-811.910265, -811.910265, -101.488783],
            decimal=6,
        )

        # |> test restart functionality
        print("==================================================")
        boo2 = Booster(None, log=True)

        # |> new show functionality
        boo2.show("TestPortfolio")

        print("==================================================")

        # |> do restart
        boo2.task_portfolio("TestPortfolio", run=True, save_to_storage=True)
        print(
            z_ld(
                "runs/BooTest/sim.0.(PORTFOLIO)/TestPortfolio_PORTFOLIO",
                dbname="booster",
            ).result.executions
        )

        # - test report after restart (test new functionality from 2023-Mar-21) -
        rep1 = z_ld("portfolios/BooTest/TestPortfolio", dbname="booster")["report"]
        print(rep1["Set0"])
        print(rep1["Set1"])
        np.testing.assert_almost_equal(
            rep1["Set0"]["Gain"].values, [703.166439, 703.166439, 87.895805], decimal=6
        )
        np.testing.assert_almost_equal(
            rep1["Set1"]["Gain"].values,
            [-811.910265, -811.910265, -101.488783],
            decimal=6,
        )
        print("##################################################")

        boo2.ls()

        print("##################################################")

        # how about the deletion
        boo.delete_previous_portfolio_runs("BooTest", "TestPortfolio")
        print(z_ls("portfolios/.*", dbname="booster"))
        self.assertEqual(z_ls("runs/.*", dbname="booster"), [])
        self.assertEqual(z_ls("stats/.*", dbname="booster"), [])
        print(z_ld("portfolios/_index/BooTest/TestPortfolio", dbname="booster"))


from pytest import main

if __name__ == "__main__":
    main()
