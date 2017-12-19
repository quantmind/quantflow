from pulsar.apps.rpc import JsonProxy


APIS = dict(
    betting=dict(
        prefix='SportsAPING/v1.0',
        methods=(
            # read-only
            'listEventTypes',
            'listCompetitions',
            'listTimeRanges',
            'listEvents',
            'listMarketTypes',
            'listCountries',
            'listVenues',
            'listMarketCatalogue',
            'listMarketBook',
            'listMarketProfitAndLoss',
            'listCurrentOrders',
            'listClearedOrders',
            # transactional
            'placeOrders',
            'cancelOrders',
            'replaceOrders',
            'updateOrders',
        ),
    ),
    account=dict(
        prefix='AccountAPING/v1.0',
        methods=(
            'getAccountFunds',
            'getAccountDetails',
        )
    ),
    scores=dict(
        prefix='ScoresAPING/v1.0',
        methods=(
            'listRaceDetails',
        )
    )
)


ERRORS = {
    'DSC-0008': dict(
        fault='JSONDeserialisationParseFailure'
    ),
    'DSC-0009': dict(
        fault='ClassConversionFailure',
        msg='Invalid value for parameter'
    ),
    'DSC-0015': dict(
        fault='SecurityException',
        status=403,
        msg='Credentials supplied in request were invalid'
    ),
    'DSC-0018': dict(
        fault='MandatoryNotDefined',
        msg='A parameter marked as mandatory was not provided'
    ),
    'DSC-0019': dict(
        fault='Timeout',
        status=504,
        msg='The request has timed out'
    ),
    'DSC-0021': dict(
        fault='NoSuchOperation',
        status=404,
        msg='The operation specified does not exist'
    ),
    'DSC-0023': dict(
        fault='NoSuchService',
        status=404
    ),
    'DSC-0024': dict(
        fault='RescriptDeserialisationFailure',
        msg='Exception during deserialization of RESCRIPT request'
    ),
    'DSC-0034': dict(
        fault='Unknown',
        msg=(
            "A valid and active App Key hasn't been provided in the request. "
            "Please check that your App Key is active. "
            "Please see Application Keys for further information "
            "regarding App Keys."
        ),
    ),
    'DSC-0035': dict(
        fault='UnrecognisedCredentials'
    ),
    'DSC-0036': dict(
        fault='InvalidCredentials'
    )
}


class BetFairError(RuntimeError):
    pass


class BetFairResponseError(BetFairError):
    fault = None

    def __init__(self, code, msg):
        self.code = code
        self.msg = msg
        error = ERRORS.get(msg)
        if error:
            self.status = error.get('status', 400)
            self.fault = error['fault']
            self.msg = '%s - %s' % (msg, self.fault)
            if 'msg' in error:
                self.msg = '%s - %s' % (self.msg, error['msg'])
        else:
            self.status = code

    def __repr__(self):
        return '%s - %s' % (self.status, self.msg)
    __str__ = __repr__


class BetFairApi(JsonProxy):
    """"A betfair JSON-RPC client
    """
    def __init__(self, url, http, headers=None, prefix=None, methods=None):
        super().__init__(url, http=http, headers=headers)
        self.prefix = prefix
        self.methods = methods

    def __getattr__(self, name):
        if name not in self.methods:
            raise BetFairError('Method %s not supported' % name)
        return super().__getattr__('%s/%s' % (self.prefix, name))

    @classmethod
    def exception(cls, code, msg):
        return BetFairResponseError(code, msg)
