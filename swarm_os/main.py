import asyncio
from swarm_os.core.agents import MicroAgent, SwarmManager

async def main():
    print("Initializing A-Swarm-OS MVP...")

    manager = SwarmManager()

    # Bootstrap: Create a simple sum function
    initial_code = """
def calculate_sum(a, b):
    # I only know how to sum simple numbers
    return a + b
"""

    # We set a low threshold to easily trigger performance degradation if needed,
    # but the primary trigger here will be Type/Value Error.
    sum_agent = MicroAgent(
        agent_id="sum_agent",
        initial_code=initial_code,
        entry_point="calculate_sum",
        threshold_ms=50.0
    )

    manager.register_agent(sum_agent)

    analyst_code = """
def analyze_sum(matrix_a, matrix_b, result):
    # A simple validation to see if we just concatenated instead of adding element-wise
    if isinstance(matrix_a, list) and isinstance(matrix_b, list):
        if len(result) == len(matrix_a) + len(matrix_b):
            # This looks like simple list concatenation
            return False
    return True
"""

    analyst_agent = MicroAgent(
        agent_id="analyst_agent",
        initial_code=analyst_code,
        entry_point="analyze_sum",
        threshold_ms=50.0
    )
    manager.register_agent(analyst_agent)


    print("\n--- Test 1: Simple Integer Sum ---")
    try:
        result = await manager.execute_agent("sum_agent", 5, 10)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\n--- Test 2: Matrix Sum (Will trigger RFI) ---")
    matrix_a = [[1, 2], [3, 4]]
    matrix_b = [[5, 6], [7, 8]]

    print(f"Input A: {matrix_a}")
    print(f"Input B: {matrix_b}")

    try:
        # Execute sum agent
        res = await manager.execute_agent("sum_agent", matrix_a, matrix_b)

        # Use analyst agent to check if the logic is sound
        is_valid = await manager.execute_agent("analyst_agent", matrix_a, matrix_b, res)

        if not is_valid:
            print("[Analyst] Detected incorrect logic (List Concatenation instead of Element-wise Addition)!")
            from swarm_os.core.agents import RFIException
            agent = manager.agents["sum_agent"]
            await manager.handle_rfi(RFIException(
                message="List concatenation detected instead of matrix addition!",
                trigger_reason="Incapacidad Logística (List Concatenation instead of Matrix Addition)",
                agent_id="sum_agent",
                code=agent.code,
                kwargs={'args': (matrix_a, matrix_b), 'error': "Analyst agent rejected the result"}
            ))
            # Retry after evolution
            res = await manager.execute_agent("sum_agent", matrix_a, matrix_b)

        result = res
        print(f"Final Result after Evolution: {result}")
    except Exception as e:
        print(f"Final Failure: {e}")

    # Save memory to persist state
    manager.memory.save()

if __name__ == "__main__":
    asyncio.run(main())
