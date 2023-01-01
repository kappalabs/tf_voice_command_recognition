import os

from typing import Tuple

from commands_processor import Command


class CommandsExecution:

    @staticmethod
    def process_sentence(sentence: Tuple[Command, Command]):
        print("processing sentence", sentence[0].name, sentence[1].name)

        if sentence[0].name == "go" and sentence[1].name == "on":
            print("go on")
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba/set\' -m \'{"brightness": 254}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba_2/set\' -m \'{"brightness": 254}\'')
        elif sentence[0].name == "go" and sentence[1].name == "off":
            print("go off")
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba/set\' -m \'{"brightness": 0}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba_2/set\' -m \'{"brightness": 0}\'')
        else:
            print("unknown sentence", sentence[0].name, sentence[1].name)
