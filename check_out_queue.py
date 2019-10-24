import asyncio
from nats.aio.client import Client as NATS


async def example():

    nc = NATS()
    await nc.connect('nats://nats-cluster-nats-headless.default.svc.cluster.local:4222')
    queue = asyncio.Queue()

    async def cb(msg):
        await queue.put(msg)

    await nc.subscribe("movementDetectionImage", cb=cb)

    while True:
        msg = await queue.get()
        print("Msg:", msg)


loop = asyncio.get_event_loop()
loop.run_until_complete(example())
loop.close()
