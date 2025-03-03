import keyring
from snowflake.snowpark.session import Session

def snowflake_connect():

    login = keyring.get_password('snowflake','account')
    getin = keyring.get_password('snowflake','sgwoods')

    connection_parameters = {
        'account':login,
        'user':'sgwoods',
        'password':getin,
        'role':'SYSADMIN',
        'warehouse':'COMPUTE_WH',
        'database':'NFL',
        'schema':'tracking'
    }

    session = Session.builder.configs(connection_parameters).create()

    return session