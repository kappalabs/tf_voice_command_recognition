from typing import Tuple

from commands_processor import Command


class CommandsExecution:

    @staticmethod
    def process_sentence(sentence: Tuple[Command, Command]):
        print("processing sentence", sentence[0].name, sentence[1].name)

        if sentence[0].name == "go" and sentence[1].name == "on":
            print("go on")
        elif sentence[0].name == "go" and sentence[1].name == "off":
            print("go off")
        else:
            print("unknown sentence", sentence[0].name, sentence[1].name)
