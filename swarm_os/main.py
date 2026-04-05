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
        # We manually trigger an RFI here by wrapping in try-catch and throwing ValueError
        # to ensure it learns matrix addition explicitly if `+` just concatenates lists
        try:
            # Check if it just concatenated
            res = await manager.execute_agent("sum_agent", matrix_a, matrix_b)
            if res == [[1, 2], [3, 4], [5, 6], [7, 8]]:
                raise ValueError("List concatenation detected instead of matrix addition!")
        except Exception as inner_e:
            from swarm_os.core.agents import RFIException
            # Manually throw RFI to force evolution
            agent = manager.agents["sum_agent"]
            await manager.handle_rfi(RFIException(
                message=str(inner_e),
                trigger_reason="Incapacidad Logística (List Concatenation instead of Matrix Addition)",
                agent_id="sum_agent",
                code=agent.code,
                kwargs={'args': (matrix_a, matrix_b), 'error': str(inner_e)}
            ))

        result = await manager.execute_agent("sum_agent", matrix_a, matrix_b)
        print(f"Final Result after Evolution: {result}")
    except Exception as e:
        print(f"Final Failure: {e}")

if __name__ == "__main__":
    asyncio.run(main())
