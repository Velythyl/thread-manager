def evaluate_agent(ppo_trainer, agent, EVAL_OUTPUT_FOLDER):
    policy = ppo_trainer.agent
    policy.ac.eval()

    with suppress_stderr():
        envs = make_vec_envs(xml_file=agent, training=False, norm_rew=False, render_policy=True)
        set_ob_rms(envs, get_ob_rms(ppo_trainer.envs))

    episode_return = evaluate(policy, envs)
    envs.close()
    print(agent, f'{episode_return.mean():.2f} +- {episode_return.std():.2f}')

    with open(f'{EVAL_OUTPUT_FOLDER}/eval_{agent}.pkl', 'wb') as f:
        pickle.dump(episode_return, f)

    return episode_return

def threadmanager_run_spoof(target, args):
    return_container = args[0]
    id = args[1]
    try:
        result = target(*(args[2:]))
    except Exception as e:
        print(e.with_traceback())
        print(e)
        import traceback
        traceback.print_exc()
    return_container.append(result)
    print(f"Thread id {id} done!")
    return

class ThreadManager:
    def __init__(self, max_proc, proc_per_thread, max_threads=None, waiting_timeout=5, batch_mode=True):
        self.threads = {}
        self.threads_return_containers = {}

        self.max_proc = max_proc
        self.proc_per_thread = proc_per_thread

        self.max_threads = max_proc // proc_per_thread
        if max_threads is not None:
            self.max_threads = min(max_threads, self.max_threads)
        self.max_threads -= 1

        self.waiting_timeout = waiting_timeout

        self.ordering = []

        self.batch_mode = batch_mode

    def _wait_for_space(self, override_max_threads=None):
        max_threads = self.max_threads
        if override_max_threads is not None:
            max_threads = override_max_threads

        ITER = 0
        while len(self.threads) > max_threads:  # waits until there's at least one free space; if max is 5 and there's 5, blocks until there's 4; is max is 0 and there's 10, waits until there's 0
            print(f"Loop iter {ITER}")
            print(f"Num active threads {len(self.threads)} out of max threads {max_threads + 1}")
            ids = list(self.threads.keys())
            for id in ids:
                if id not in self.threads:
                    continue

                if not self.threads[id].is_alive():
                    print(f"{id} is done, joining...")
                    self.threads[id].join()
                    print(f"...joined {id}")
                    assert len(self.threads_return_containers[id]) == 1
                    del self.threads[id]
            time.sleep(self.waiting_timeout)
            ITER += 1

    def _uuidgen(self):
        return str(uuid.uuid4().hex)

    def task(self, target, args):
        return_container = []
        id = self._uuidgen()
        thread = threading.Thread(target=threadmanager_run_spoof, args=(target, (return_container,id, *args)))
        thread.daemon = True

        if not self.batch_mode:
            self._wait_for_space()
            thread.start()

        self.threads[id] = thread
        self.threads_return_containers[id] = return_container
        self.ordering.append(id)

        if self.batch_mode and len(self.threads) >= self.max_threads:
            print(f"Starting batch of {len(self.threads)} threads...")
            for t in self.threads.values():
                t.start()
            self._wait_for_space(override_max_threads=0)
            print(f"...finished batch of {len(self.threads)} threads.")

    def get_results(self):
        self._wait_for_space(override_max_threads=0) # wait for all threads
        return [self.threads_return_containers[id][0] for id in self.ordering]
