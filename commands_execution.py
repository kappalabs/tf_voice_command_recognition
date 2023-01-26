import os
import time

from typing import Tuple

from commands_processor import Command


class CommandsExecution:

    @staticmethod
    def process_sentence(sentence: Tuple[Command, Command]):
        print("{}: processing sentence".format(time.strftime("%H:%M:%S %d.%m.%Y")), sentence[0].name, sentence[1].name)

        if sentence[0].name == "rozsvit" and sentence[1].name == "chodba":
            os.system('mosquitto_pub -t \'hlas/chodba/svetla/on\' -m \'{}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "chodba":
            os.system('mosquitto_pub -t \'hlas/chodba/svetla/off\' -m \'{}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "pokoj":
            os.system('mosquitto_pub -t \'hlas/pokoj/svetla/on\' -m \'{}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "pokoj":
            os.system('mosquitto_pub -t \'hlas/pokoj/svetla/off\' -m \'{}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "postel":
            os.system('mosquitto_pub -t \'hlas/loznice/svetla/on\' -m \'{}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "postel":
            os.system('mosquitto_pub -t \'hlas/loznice/svetla/off\' -m \'{}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "koupelna":
            os.system('mosquitto_pub -t \'hlas/koupelna/svetla/on\' -m \'{}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "koupelna":
            os.system('mosquitto_pub -t \'hlas/koupelna/svetla/off\' -m \'{}\'')
        elif sentence[0].name == "rozsvit" and sentence[1].name == "kuchyn":
            os.system('mosquitto_pub -t \'hlas/kuchyn/svetla/on\' -m \'{}\'')
        elif sentence[0].name == "zhasni" and sentence[1].name == "kuchyn":
            os.system('mosquitto_pub -t \'hlas/kuchyn/svetla/off\' -m \'{}\'')
        else:
            print("unknown sentence", sentence[0].name, sentence[1].name)
