TOL = 50  # => per cent
DELTA = 1.0e-12


def get_ape_from_pd_series(quantum_pd_series, classical_pd_series):
    """Helper function to evaluate absolute percentage error between classical and quantum results."""
    ape = (
        abs(quantum_pd_series - classical_pd_series)
        * 100.0
        / abs(classical_pd_series + DELTA)
    )
    return ape


def compare_results(classical_result, quantum_result):
    """Helper function that compares the classical and quantum simulation results."""
    classical_data = []
    quantum_data = []

    def check_ape(classical_value, quantum_value):
        """Helper function to check if the absolute percentage error between classical and quantum results is within TOL."""  # noqa: E501
        ape = (
            abs(quantum_value - classical_value) * 100.0 / abs(classical_value + DELTA)
        )
        is_close_to_classical = ape <= TOL
        if is_close_to_classical:
            print(
                f"Quantum result {quantum_value} within {ape}% of classical result {classical_value}"
            )
            quantum_data.append(quantum_value)
            classical_data.append(classical_value)
        return is_close_to_classical

    for link in classical_result.link["flowrate"].columns:
        classical_value = classical_result.link["flowrate"][link].iloc[0]
        quantum_value = quantum_result.link["flowrate"][link].iloc[0]
        message = f"Flowrate {link}: {quantum_value} not within {TOL}% of classical result {classical_value}"
        assert check_ape(classical_value, quantum_value), message

    for node in classical_result.node["pressure"].columns:
        classical_value = classical_result.node["pressure"][node].iloc[0]
        quantum_value = quantum_result.node["pressure"][node].iloc[0]
        message = f"Pressure {node}: {quantum_value} not within {TOL}% of classical result {classical_value}"
        assert check_ape(classical_value, quantum_value), message

    return classical_data, quantum_data  # noqa: W292
