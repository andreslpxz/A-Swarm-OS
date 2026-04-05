import time
import inspect
import asyncio
from typing import Callable, Any, Dict, List, Optional
from swarm_os.core.llm import GroqClientWrapper
from swarm_os.core.fractal_memory import FractalGraphMemory

class RFIException(Exception):
    def __init__(self, message: str, trigger_reason: str, agent_id: str, code: str, kwargs: Dict[str, Any]):
        super().__init__(message)
        self.trigger_reason = trigger_reason
        self.agent_id = agent_id
        self.code = code
        self.kwargs = kwargs

class MicroAgent:
    def __init__(self, agent_id: str, initial_code: str, entry_point: str, threshold_ms: float = 100.0):
        self.agent_id = agent_id
        self.code = initial_code
        self.entry_point = entry_point
        self.threshold_ms = threshold_ms
        self._compiled_func = None
        self._namespace = {}
        self._compile()

    def _compile(self):
        self._namespace = {}
        try:
            exec(self.code, self._namespace)
            self._compiled_func = self._namespace.get(self.entry_point)
            if not self._compiled_func:
                raise ValueError(f"Entry point {self.entry_point} not found in code")
        except Exception as e:
            raise RuntimeError(f"Failed to compile agent {self.agent_id}: {e}")

    def hot_swap(self, new_code: str):
        print(f"[{self.agent_id}] Hot-swapping code...")
        old_code = self.code
        self.code = new_code
        try:
            self._compile()
            print(f"[{self.agent_id}] Hot-swap successful.")
        except Exception as e:
            print(f"[{self.agent_id}] Hot-swap failed, reverting. Error: {e}")
            self.code = old_code
            self._compile()

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()

        try:
            result = self._compiled_func(*args, **kwargs)
        except (TypeError, ValueError) as e:
            raise RFIException(
                message=f"Agent failed execution: {e}",
                trigger_reason="Incapacidad Logística",
                agent_id=self.agent_id,
                code=self.code,
                kwargs={'args': args, 'kwargs': kwargs, 'error': str(e)}
            )

        end_time = time.perf_counter()
        execution_ms = (end_time - start_time) * 1000

        if execution_ms > self.threshold_ms:
            raise RFIException(
                message=f"Agent execution too slow: {execution_ms:.2f}ms (threshold: {self.threshold_ms}ms)",
                trigger_reason="Degradación de Performance",
                agent_id=self.agent_id,
                code=self.code,
                kwargs={'args': args, 'kwargs': kwargs, 'execution_ms': execution_ms}
            )

        return result

class SwarmManager:
    def __init__(self):
        self.agents: Dict[str, MicroAgent] = {}
        self.llm = GroqClientWrapper()
        self.memory = FractalGraphMemory()

    def register_agent(self, agent: MicroAgent):
        self.agents[agent.agent_id] = agent
        self.memory.add_node(
            id=f"agent_{agent.agent_id}",
            content=f"Agent: {agent.agent_id}\nCode:\n{agent.code}",
            metadata={"agent_id": agent.agent_id, "type": "agent_registry"}
        )

    async def handle_rfi(self, rfi: RFIException):
        print(f"\n[SwarmManager] Received RFI from '{rfi.agent_id}'. Reason: {rfi.trigger_reason}")
        print(f"[SwarmManager] Details: {rfi.kwargs}")

        agent = self.agents.get(rfi.agent_id)
        if not agent:
            print(f"[SwarmManager] Error: Agent {rfi.agent_id} not found.")
            return

        # Prepare memory context
        relevant_memories = self.memory.retrieve_relevant(f"How to fix {rfi.trigger_reason} for agent {rfi.agent_id}", top_k=2)
        memory_context = "\n".join([f"- {m.content}" for m in relevant_memories])

        system_prompt = """You are a Senior AI Architect and Expert in Evolutionary Computation.
Your task is to refactor the provided Python code to fix the issue described.
You must return ONLY the raw Python code, without markdown blocks (```python ... ```), explanations, or any other text.
The code must contain a function with the same entry point name as the original.
Optimize for extreme speed, vectorization (if using lists/matrices, consider standard Python zip/list comprehensions if numpy isn't imported in the code), and handle the error case properly.
Important: If the issue is related to adding matrices or lists together element-wise, you MUST add them element-wise, not just concatenate them! For matrices (lists of lists), you must return a new list of lists where each element is the sum of the corresponding elements in the input matrices. e.g. [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]
Si necesitas mayor eficiencia, puedes importar librerías como numpy o math dentro de la función.
"""

        prompt = f"""
Agent ID: {rfi.agent_id}
Entry point: {agent.entry_point}
Trigger Reason: {rfi.trigger_reason}
Details: {rfi.kwargs}

Current Code:
{agent.code}

Relevant Memory Context:
{memory_context}

Refactor the code to handle the issue and improve performance. Return ONLY the python code.
"""

        print(f"[SwarmManager] Requesting Refactor from Groq...")
        new_code_raw = await self.llm.generate(prompt=prompt, system_prompt=system_prompt)

        # Clean up code if Groq accidentally added markdown
        new_code = new_code_raw.replace("```python", "").replace("```", "").strip()

        print(f"[SwarmManager] Refactored code received:\n{'-'*20}\n{new_code}\n{'-'*20}")

        # Hot swap
        agent.hot_swap(new_code)

        # Update memory
        self.memory.add_node(
            id=f"rfi_{time.time()}",
            content=f"Refactored agent {agent.agent_id} due to {rfi.trigger_reason}. New code: {new_code}",
            metadata={"agent_id": agent.agent_id, "type": "rfi_resolution"}
        )

    async def execute_agent(self, agent_id: str, *args, **kwargs):
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        try:
            return agent(*args, **kwargs)
        except RFIException as rfi:
            await self.handle_rfi(rfi)
            # Retry execution after hot-swap
            print(f"[SwarmManager] Retrying execution of '{agent_id}' after hot-swap...")
            return agent(*args, **kwargs)
