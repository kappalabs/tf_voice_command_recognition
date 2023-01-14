import time


class Command:

    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence
        self.timestamp = time.time()

    def __str__(self):
        return f'{self.name} ({self.confidence})'


class CommandsProcessor:

    def __init__(self, threshold_confidence=0.98, threshold_time_sec=2):
        self.threshold = threshold_confidence
        self.threshold_time_sec = threshold_time_sec

        self.commands = []

    def remove_old_commands(self, timestamp):
        self.commands = [command for command in self.commands
                         if command.timestamp > timestamp - self.threshold_time_sec]

    def add_command(self, new_command: Command):
        appended = False
        # filtered_command_names = set()
        filtered_commands = dict()
        for command in self.commands:
            # Select the command with the largest confidence
            if command.name == new_command.name:
                if command.confidence > new_command.confidence:
                    filtered_commands[command.name] = command
                else:
                    filtered_commands[new_command.name] = new_command
                    appended = True
            else:
                filtered_commands[command.name] = command

        if not appended:
            filtered_commands[new_command.name] = new_command

        self.commands = list(filtered_commands.values())

    def match_command_tuple(self, new_command: Command):
        # Find command with the largest confidence
        max_confidence = 0
        max_command = None
        for command in self.commands:
            if command.confidence > max_confidence and command.name != new_command.name:
                max_confidence = command.confidence
                max_command = (command, new_command)

        return max_command

    def process(self, command: Command):
        # Determine if the command should be saved
        if command.confidence < self.threshold:
            return None

        # Clear old commands
        self.remove_old_commands(command.timestamp)

        # Check if there is a known command sentence/tuple
        sentence = self.match_command_tuple(command)

        print(command.name, command.confidence)
        print([str(command) for command in self.commands])

        # If there is no known command sentence/tuple, add the new command
        if sentence is None:
            # Add the command
            self.add_command(command)
        else:
            # Clear the commands
            self.commands = []

        return sentence
