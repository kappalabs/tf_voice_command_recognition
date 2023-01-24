import os

from typing import Tuple

from commands_processor import Command


class CommandsExecution:

    @staticmethod
    def process_sentence(sentence: Tuple[Command, Command]):
        print("processing sentence", sentence[0].name, sentence[1].name)

        if sentence[0].name == "rozsvit" and sentence[1].name == "chodba":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba_1/set\' -m \'{"state": "on"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba_2/set\' -m \'{"state": "on"}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "chodba":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba_1/set\' -m \'{"state": "off"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_chodba_2/set\' -m \'{"state": "off"}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "pokoj":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_1/set\' -m \'{"state": "on"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_2/set\' -m \'{"state": "on"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_3/set\' -m \'{"state": "on"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_4/set\' -m \'{"state": "on"}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "pokoj":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_1/set\' -m \'{"state": "off"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_2/set\' -m \'{"state": "off"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_3/set\' -m \'{"state": "off"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_pokoj_4/set\' -m \'{"state": "off"}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "postel":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_1100_1/set\' -m \'{"state": "on"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_1100_2/set\' -m \'{"state": "on"}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "postel":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_1100_1/set\' -m \'{"state": "off"}\'')
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_1100_2/set\' -m \'{"state": "off"}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "koupelna":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_koupelna/set\' -m \'{"state": "on"}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "koupelna":
            os.system('mosquitto_pub -t \'zigbee2mqtt/svetlo_koupelna/set\' -m \'{"state": "off"}\'')
        else:
            print("unknown sentence", sentence[0].name, sentence[1].name)
