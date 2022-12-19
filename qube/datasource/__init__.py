# Here global lookup table where we store connector names
_CONNECTORS_LOOKUP = {
    'csv': 'CsvConnector',
    'yahoo': 'YahooConnector',
    'kdb': 'KdbConnector',
    'kdb_connector': 'KdbConnector',
    'dukas_outlook': 'DukasOutlookConnector',
    'mongo_connector': 'MongoConnector'
}

from .DataSource import *
