"""Backend server
"""
from pq.api import PulsarQueue


def app():
    return PulsarQueue(
        consumers=[
            'pq.api:Tasks',
            'providers.truefx:Aggregator'
        ],
        description='Pulsar queue with twitter streaming'
    )


if __name__ == '__main__':
    app().start()
