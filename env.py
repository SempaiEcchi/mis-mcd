import simpy
import random
import numpy as np


class Order:
    ITEM_NAMES = ['hamburger', 'drink', 'ice_cream']

    def __init__(self, id, items, arrival_time, num_people=1):
        self.id = id
        self.items = list(items)
        self.arrival_time = arrival_time
        self.num_people = num_people

    def num_items(self):
        return len(self.items)

    def contains(self, typ):
        return typ in self.items


class McDonaldsEnv:
    def __init__(self, seed=None, sim_time=8 * 60, arrival_rate=1 / 3.0, max_active_orders=20, serve_window=3):

        self.sim_time = sim_time
        self.arrival_rate = arrival_rate
        self.max_active_orders = max_active_orders
        self.serve_window = serve_window
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.env = simpy.Environment()

        self.queue = []
        self.next_id = 0
        self.server_busy = False
        self.current_time = 0.0
        self.wait_times = []

        self.served_orders = 0
        self.served_items = 0
        self.total_wait = 0.0
        self.served_people = 0
        self.total_wait_people = 0.0

        self.env.process(self._arrival_process())
        return self._get_state()

    def _arrival_process(self):
        while True:
            minute_in_cycle = self.env.now % 120
            if minute_in_cycle < 60:
                current_arrival_rate = self.arrival_rate * 1.5
            else:
                current_arrival_rate = self.arrival_rate
            inter = self.rng.expovariate(current_arrival_rate)
            yield self.env.timeout(inter)
            self.current_time = self.env.now
            items = self._sample_order()
            n_items = len(items)
            if n_items == 1:
                num_people = self.rng.choices([1,2], weights=[0.9,0.1])[0]
            elif n_items == 2:
                num_people = self.rng.choices([1,2,3], weights=[0.7,0.25,0.05])[0]
            elif n_items <= 4:
                num_people = self.rng.choices([2,3,4], weights=[0.4,0.4,0.2])[0]
            elif n_items <= 6:
                num_people = self.rng.choices([2,3,4,5], weights=[0.2,0.4,0.3,0.1])[0]
            else:
                num_people = self.rng.choices([2,3,4,5,6], weights=[0.1,0.2,0.3,0.3,0.1])[0]
            if len(self.queue) >= self.max_active_orders:
                if not hasattr(self, 'rejected_orders'):
                    self.rejected_orders = 0
                self.rejected_orders += 1
                continue
            order = Order(self.next_id, items, self.env.now, num_people=num_people)
            self.next_id += 1
            self.queue.append(order)

    def _sample_order(self):
        # Allow 1-8 items, most orders are small, some are large
        size = self.rng.choices([1,2,3,4,5,6,7,8], weights=[0.3,0.25,0.15,0.1,0.08,0.06,0.04,0.02])[0]
        probs = [0.5, 0.3, 0.2]
        items = [self.rng.choices([0, 1, 2], weights=probs)[0] for _ in range(size)]
        return items

    def _service_time(self, typ):

        base = 1.0 if typ == 0 else 0.1
        noise = self.rng.uniform(-0.05, 0.05) * base
        return max(0.01, base + noise)

    def _get_state(self):

        counts = [0, 0, 0]
        waits = [0.0, 0.0, 0.0]
        now = self.env.now if hasattr(self, 'env') else 0.0
        order_counts = [0, 0, 0]

        for order in self.queue:
            for typ in order.items:
                counts[typ] += 1
                waits[typ] += (now - order.arrival_time)
            for typ in set(order.items):
                order_counts[typ] += 1

        avg_waits = [(waits[i] / order_counts[i]) if order_counts[i] > 0 else 0.0 for i in range(3)]
        state = np.array(counts + avg_waits, dtype=np.float32)
        return state

    def _eligible_indices(self):
        return list(range(len(self.queue)))

    def serve_order_index(self, idx):

        if not self.queue:
            if self.env.peek() is None or self.env.now >= self.sim_time:
                return self._get_state(), 0.0, True, {}
            self.env.step()
            return self._get_state(), 0.0, False, {}

        eligible = self._eligible_indices()
        if not eligible:
            return self._get_state(), 0.0, self.env.now >= self.sim_time, {}

        if idx < 0 or idx >= len(self.queue) or idx not in eligible:
            idx = eligible[0]
        order = self.queue.pop(idx)
        arrival_time = order.arrival_time
        service_time = sum(self._service_time(typ) for typ in order.items)
        target = self.env.now + service_time
        while self.env.peek() is not None and self.env.peek() < target:
            self.env.step()
        self.env.run(until=target)
        wait_per_order = (self.env.now - arrival_time - service_time)
        self.wait_times.append(wait_per_order)
        self.served_orders += 1
        self.served_items += order.num_items()
        self.total_wait += wait_per_order
        self.served_people += order.num_people
        self.total_wait_people += wait_per_order * order.num_people
        done = self.env.now >= self.sim_time


        eligible = self._eligible_indices()
        spt_idx = None
        spt_est = float('inf')
        for i in eligible:
            o = self.queue[i] if i < len(self.queue) else order
            est = sum(self._service_time(t) for t in o.items)
            if est < spt_est:
                spt_est = est
                spt_idx = i

        if spt_idx is not None:
            spt_order = self.queue[spt_idx] if spt_idx < len(self.queue) else order
            spt_service_time = sum(self._service_time(t) for t in spt_order.items)
            spt_wait = (self.env.now - spt_order.arrival_time - spt_service_time)
        else:
            spt_wait = wait_per_order

        reward = 2.0 * (spt_wait - wait_per_order)
        reward -= 0.5 * wait_per_order

        #Penalty for number of people waiting after serving
        people_waiting = sum(o.num_people for o in self.queue)
        people_penalty = 0.05 * people_waiting
        reward -= people_penalty


        people_bonus = 0.2 * (order.num_people - 1)
        reward += people_bonus

        reward = np.clip(reward, -5, 5)
        return self._get_state(), reward, done, {}

    def step(self, action):

        if not self.queue:

            if self.env.peek() is None or self.env.now >= self.sim_time:
                done = True
                return self._get_state(), 0.0, done, {}

            self.env.step()
            return self._get_state(), 0.0, False, {}

        eligible = self._eligible_indices()
        idx = None
        for i in eligible:
            if self.queue[i].contains(action):
                idx = i
                break
        if idx is None:
            idx = eligible[0]
        return self.serve_order_index(idx)

    def run_random_episode(self, max_steps=1000):
        self.reset()
        total_reward = 0.0
        steps = 0
        while True and steps < max_steps:
            state = self._get_state()
            if all(x == 0 for x in state[:3]):
                if self.env.peek() is None or self.env.now >= self.sim_time:
                    break
                self.env.step()
                continue

            eligible = self._eligible_indices()
            if not eligible:
                if self.env.peek() is None or self.env.now >= self.sim_time:
                    break
                self.env.step()
                continue
            idx = self.rng.choice(eligible)
            _, r, done, _ = self.serve_order_index(idx)
            total_reward += r
            steps += 1
            if done:
                break
        avg_wait = (self.total_wait_people / self.served_people) if self.served_people > 0 else 0.0
        return total_reward, avg_wait
