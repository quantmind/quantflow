var casper = require('casper'),
    utils = require('utils'),
    system = require('system'),
    username = system.env.USERNAME,
    password = system.env.PASSWORD,
    day = system.env.DAY,
    month = system.env.MONTH,
    year = system.env.YEAR,
    URL = "https://selftrade.co.uk/",
    TIMEOUT = 30000,
    SKIP_LOG = ['media', 'fonts', 'bundles'];


var cli = casper.create({
    pageSettings: {
        userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:23.0) Gecko/20130404 Firefox/23.0"
    }
});


cli.on('resource.received', function (resource) {
    if (log(resource.url)) {
        resource.body = this.getPageContent();
        this.echo(utils.dump(resource));
    }
});

cli.on('resource.requested', function (resource, request) {
    // if (log(resource.url))
    //     this.echo(resource.method + ' ' + resource.url);
});

cli.start(URL + "transactional/anonymous/login");
cli.waitForSelector('form#LoginForm', preLogin);
cli.waitUntilVisible('label#PasswordCharacter1Label', login);
cli.waitUntilVisible('div#Dashboard', accountsInfo, null, TIMEOUT);
cli.then(logout);
cli.run();



function log (url) {
    if (url.substring(0, URL.length) === URL) {
        var bits = url.substring(URL.length).split('/');
        return (SKIP_LOG.indexOf(bits[0]) === -1);
    }
    return false;
}


function preLogin () {
    this.fill('form#LoginForm', {
        'Username': username,
        'DateOfBirthViewModel.Day': day,
        'DateOfBirthViewModel.Month': month,
        'DateOfBirthViewModel.Year': year
    });
    this.click('#PreLoginButton');
}


function login () {
    var self = this;
        data = {};

    [1, 2, 3].map(function (n) {
        var id = 'label#PasswordCharacter' + n + 'Label strong';
        var text = self.getElementInfo(id).text,
            idx = +text.substring(0, text.length-2);
        data['PasswordCharacter' + n] = password.substring(idx-1, idx);
    });
    this.fill('form#LoginForm', data);
    this.click('#LoginButton');
}

function logout () {
    this.click('div.logOut a');
}

function accountsInfo () {
}
