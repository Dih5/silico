import os
import sys
import importlib
import curses

from .base import Experiment

# Constants for positioning
EXPERIMENT_LIST_START = 2
DETAILS_START = 2
OUTPUT_START = 13  # Starting line for output messages
FOOTER_OFFSET = 2  # How many lines after the last output to display the footer message


def get_experiments(file):
    """Get all experiments from a file."""
    if file.lower().endswith(".py"):
        file = file[:-3]

    # Ensure the cwd is in the path
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)

    try:
        m = importlib.import_module(file)
    except ImportError:
        return None, "Error: Couldn't load the module."

    candidates = {k: e for k, e in m.__dict__.items() if isinstance(e, Experiment)}
    if len(candidates) == 0:
        return None, "Error: No experiments found in the module."
    
    return candidates, None




def run_cui(module_name):
    experiments, error = get_experiments(module_name)

    if error:
        print(error)
        return

    experiment_list = list(experiments.keys())
    selected_idx = 0  # Tracks the selected experiment in the list

    def draw_menu(stdscr):

        nonlocal selected_idx

        # Clear screen and set up basic layout
        stdscr.clear()
        curses.curs_set(0)

        # Dimensions for the layout
        height, width = stdscr.getmaxyx()
        left_width = width // 3  # Left column width

        # Title
        stdscr.addstr(0, 0, "Experiments", curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.addstr(0, left_width + 2, "Details", curses.A_BOLD | curses.A_UNDERLINE)

        # List experiments in the left column
        for idx, exp_name in enumerate(experiment_list):
            if idx == selected_idx:
                stdscr.addstr(idx + EXPERIMENT_LIST_START, 0, exp_name, curses.A_REVERSE)
            else:
                stdscr.addstr(idx + EXPERIMENT_LIST_START, 0, exp_name)

        # Display experiment details in the right block
        selected_experiment = experiment_list[selected_idx]
        stdscr.addstr(DETAILS_START, left_width + 2, f"Selected: {selected_experiment}")
        stdscr.addstr(DETAILS_START + 2, left_width + 2, "[s] Status")
        stdscr.addstr(DETAILS_START + 3, left_width + 2, "[e] Export Results")
        stdscr.addstr(DETAILS_START + 4, left_width + 2, "[r] Run Experiment")
        stdscr.addstr(DETAILS_START + 5, left_width + 2, "[d] Delete Results")
        stdscr.addstr(DETAILS_START + 7, left_width + 2, "[q] Exit")
        stdscr.addstr(DETAILS_START + 9, left_width + 2, "Use UP/DOWN to navigate and select experiment.")

        stdscr.refresh()

    def draw_progress_bar(stdscr, line, total, done, errors, width=40):
        """Draw a progress bar showing errored trials in red, done trials in green, and remaining trials as empty space."""
        if total == 0:
            return  # Avoid division by zero

        # Calculate fractions
        error_fraction = errors / total
        done_fraction = (done - errors) / total
        remaining_fraction = 1 - (done / total)

        # Number of cells to display for each portion
        error_cells = int(width * error_fraction)
        done_cells = int(width * done_fraction)
        remaining_cells = width - (error_cells + done_cells)

        # Draw the bar
        bar = ""
        bar += "█" * error_cells  # Red block for errored trials
        bar += "█" * done_cells  # Green block for completed trials
        bar += "█" * remaining_cells  # Empty space for remaining trials

        # Add the colored sections in curses
        stdscr.addstr(line, 0, bar[:error_cells], curses.color_pair(1))  # Red for errors
        stdscr.addstr(line, error_cells, bar[error_cells:error_cells + done_cells], curses.color_pair(2))  # Green for done
        stdscr.addstr(line, error_cells + done_cells, bar[error_cells + done_cells:], curses.color_pair(3))  # Grey for remaining


    def wait_for_key(stdscr, line, message="Press any key to continue..."):
        """Displays a message and waits for any key press."""
        stdscr.addstr(line, 0, message, curses.A_BOLD | curses.A_UNDERLINE)
        stdscr.refresh()
        stdscr.getch()

    def get_input(stdscr, prompt, line):
        """Prompt the user for input in curses."""
        curses.echo()
        stdscr.addstr(line, 0, prompt)
        stdscr.refresh()
        user_input = stdscr.getstr(line, len(prompt)).decode("utf-8")
        curses.noecho()
        return user_input

    def export_results(experiment, output_file):
        """Handles export logic based on the provided file name."""
        extensions = {"pkl", "tex", "csv", "json"}
        extension = output_file.rsplit(".", 1)[-1].lower()

        if extension not in extensions:
            return f"Invalid extension. Available options are: {', '.join(extensions)}."

        df = experiment.get_results_df()
        if extension == "pkl":
            df.to_pickle(output_file)
        elif extension == "tex":
            df.to_latex(output_file)
        elif extension == "csv":
            df.to_csv(output_file)
        elif extension == "json":
            df.to_json(output_file, indent=1)
        return f"Results saved to {output_file}."

    def handle_user_input(stdscr):

        # Initialize color pairs for the progress bar
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)

        nonlocal selected_idx
        curses.curs_set(0)

        while True:
            draw_menu(stdscr)

            key = stdscr.getch()

            selected_experiment = experiment_list[selected_idx]
            e = experiments[selected_experiment]

            # Handle arrow key navigation
            if key == curses.KEY_UP and selected_idx > 0:
                selected_idx -= 1
            elif key == curses.KEY_DOWN and selected_idx < len(experiment_list) - 1:
                selected_idx += 1
            elif key == ord('q'):  # Quit the interface
                break
            elif key == ord('s'):  # Status check
                d = e.status()

                # Draw progress bar
                draw_progress_bar(stdscr, OUTPUT_START, d['total'], d['done'], d['errors'])

                # Status message
                status_message = f"{d['done']}/{d['total']} ({d['done'] / d['total'] * 100:.2f}%) trials done."
                if d['errors']:
                    status_message += f" {d['errors']}/{d['done']} ({d['errors'] / d['done'] * 100:.2f}%) errors found."
                else:
                    status_message += " No errors found."
                stdscr.addstr(OUTPUT_START + 1, 0, status_message)
                wait_for_key(stdscr, OUTPUT_START + 2 + FOOTER_OFFSET)

            elif key == ord('e'):  # Export results
                output_file = get_input(stdscr, "Enter output file name (with extension): ", OUTPUT_START)
                if output_file:
                    result_message = export_results(e, output_file)
                    stdscr.addstr(OUTPUT_START + 2, 0, result_message)
                    wait_for_key(stdscr, OUTPUT_START + 3 + FOOTER_OFFSET)
            elif key == ord('r'):  # Run experiment
                e.run_all()
                stdscr.addstr(OUTPUT_START, 0, f"Experiment '{selected_experiment}' is running...")
                wait_for_key(stdscr, OUTPUT_START + FOOTER_OFFSET)
            elif key == ord('d'):  # Delete results (invalidate)
                stdscr.addstr(OUTPUT_START, 0, "Are you sure you want to delete the results? (y/n)")
                stdscr.refresh()
                confirm_key = stdscr.getch()
                if confirm_key == ord('y'):
                    e.invalidate()
                    stdscr.addstr(OUTPUT_START + 2, 0, f"Results for '{selected_experiment}' deleted.")
                    wait_for_key(stdscr, OUTPUT_START + 3 + FOOTER_OFFSET)
                else:
                    stdscr.addstr(OUTPUT_START + 2, 0, "Deletion canceled.")
                    wait_for_key(stdscr, OUTPUT_START + 3 + FOOTER_OFFSET)

    # Initialize the curses window
    curses.wrapper(handle_user_input)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a module as an argument.")
        sys.exit(1)

    module_name = sys.argv[1]
    run_cui(module_name)
