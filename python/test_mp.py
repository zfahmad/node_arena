import multiprocessing as mp
import time
import random
import itertools


# ------------------------
# Inference process
# ------------------------
def inference_loop(request_q, response_q, shutdown_event):
    print(f"[Inference] pid={mp.current_process().pid}")

    while not shutdown_event.is_set():
        batch = []

        try:
            item = request_q.get(timeout=0.1)
        except:
            continue

        batch.append(item)

        # collect small batch
        while len(batch) < 1:
            try:
                batch.append(request_q.get_nowait())
            except:
                break

        # process batch
        req_ids = [req_id for req_id, _ in batch]
        states = [state for _, state in batch]

        time.sleep(0.2)  # simulate NN
        results = [s * s for s in states]

        for req_id, result in zip(req_ids, results):
            response_q.put((req_id, result))

    print("[Inference] shutdown")


# ------------------------
# Actor process
# ------------------------
def actor_loop(actor_id, request_q, response_q):
    print(f"[Actor {actor_id}] pid={mp.current_process().pid}")

    # local counter for request IDs
    req_counter = itertools.count(actor_id * 1000)

    for step in range(5):
        state = actor_id * 10 + step
        req_id = next(req_counter)

        # send request
        request_q.put((req_id, state))

        # wait for response
        while True:
            resp_id, result = response_q.get()
            if resp_id == req_id:
                break
            else:
                # not our response → put back
                response_q.put((resp_id, result))
                time.sleep(0.001)

        print(f"[Actor {actor_id}] state={state}, result={result}")
        time.sleep(random.uniform(0.1, 0.3))

    print(f"[Actor {actor_id}] finished")


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")

    NUM_ACTORS = 3

    request_q = mp.Queue()
    response_q = mp.Queue()
    shutdown_event = mp.Event()

    # start inference
    infer_proc = mp.Process(
        target=inference_loop,
        args=(request_q, response_q, shutdown_event),
    )
    infer_proc.start()
    actor_loop(0, request_q, response_q)

    # start actors
    # actor_procs = [
    #     mp.Process(
    #         target=actor_loop,
    #         args=(i, request_q, response_q),
    #     )
    #     for i in range(NUM_ACTORS)
    # ]
    #
    # for p in actor_procs:
    #     p.start()
    # for p in actor_procs:
    #     p.join()
    #
    # shutdown_event.set()
    # infer_proc.join()
    #
    # print("All done.")

