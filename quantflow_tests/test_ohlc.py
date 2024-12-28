from quantflow.sp.weiner import WeinerProcess
from quantflow.ta.ohlc import OHLC


def test_ohlc() -> None:
    ohlc = OHLC(
        serie="0",
        period="10m",
        parkinson_variance=True,
        garman_klass_variance=True,
        rogers_satchell_variance=True,
    )
    assert ohlc.serie == "0"
    assert ohlc.period == "10m"
    assert ohlc.index_column == "index"
    assert ohlc.parkinson_variance is True
    assert ohlc.garman_klass_variance is True
    assert ohlc.rogers_satchell_variance is True
    assert ohlc.percent_variance is False
    # create a dataframe
    path = WeinerProcess(sigma=0.5).sample(1, 1, 1000)
    df = path.as_datetime_df().reset_index()
    result = ohlc(df)
    assert result.shape == (145, 9)
