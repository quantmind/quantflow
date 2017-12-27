import tables as tb


class Quote(tb.IsDescription):
    update_type = tb.UInt8Col()
    '''Order, Trade, Cancellation, ...'''
    side = tb.UInt8Col()
    '''Bid/Buy or Ask/Sell'''
    price = tb.Float32Col()
    size = tb.Float32Col()
    timestamp = tb.Time32Col()


class Trade(Quote):
    id = tb.UInt16Col()
    # Update type indicates trade type
    update_type = tb.UInt8Col()
    # Timestamp as POSIX's time_t equivalent
    timestamp = tb.Time64Col()
    # Price
    price = tb.Float64Col()
    # Size
    size = tb.UInt32Col()
    # Indicates trade type (manual or automatic, on or off book/exchange)
    type = tb.UInt8Col()

