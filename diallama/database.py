import json
import random
import copy
import os
import re
from typing import Text, Dict

from fuzzywuzzy import fuzz
from datetime import datetime


class MultiWOZDatabase:
    """ MultiWOZ database implementation. """

    IGNORE_VALUES = {
        'hospital': ['id'],
        'police': ['id'],
        'attraction': ['location', 'openhours'],
        'hotel': ['location', 'price'],
        'restaurant': ['location', 'introduction']
    }

    FUZZY_KEYS = {
        'hospital': {'department'},
        'hotel': {'name'},
        'attraction': {'name'},
        'restaurant': {'name', 'food'},
        'bus': {'departure', 'destination'},
        'train': {'departure', 'destination'},
        'police': {'name'}
    }

    DOMAINS = [
        'restaurant',
        'hotel',
        'attraction',
        'train',
        'taxi',
        'police',
        'hospital'
    ]

    def __init__(self):
        self.data, self.data_keys = self._load_data()

    def _load_data(self):
        database_data = {}
        database_keys = {}

        for domain in self.DOMAINS:
            with open(os.path.join(os.path.dirname(__file__), "database", f"{domain}_db.json"), "r") as f:
                for l in f:
                    if not l.startswith('##') and l.strip() != "":
                        f.seek(0)
                        break
                database_data[domain] = json.load(f)

            if domain in self.IGNORE_VALUES:
                for i in database_data[domain]:
                    for ignore in self.IGNORE_VALUES[domain]:
                        if ignore in i:
                            i.pop(ignore)

            database_keys[domain] = set()
            if domain == 'taxi':
                database_data[domain] = {k.lower(): v for k, v in database_data[domain].items()}
                database_keys[domain].update([k.lower() for k in database_data[domain].keys()])
            else:
                for i, database_item in enumerate(database_data[domain]):
                    database_data[domain][i] = {k.lower(): v for k, v in database_item.items()}
                    database_keys[domain].update([k.lower() for k in database_item.keys()])

        return database_data, database_keys

    def time_str_to_minutes(self, time_string) -> Text:
        # TODO: implement the conversion
        """ Converts time to the only format supported by database, i.e. HH:MM in 24h format
            For example: "noon" -> 12:00
        """
        daytime_to_num = {"morning": "09:00", "noon": "12:00", "afternoon": "15:00", "evening": "18:00",
                          "night": "21:00", "midnight": "00:00", "breakfast": "08:00", "lunch": "13:00",
                          "dinner": "19:00", "supper": "19:00"}
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                       "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                       "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
                       "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
                       "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
                       "fifty": "50"}
        time_pattern = r"(2[0-3]|[01]?[0-9]):([0-5][0-9])"
        dot_time_pattern = r"^([01]?[0-9]|2[0-3])\.[0-5][0-9]$"
        zz_tail = ":00"

        time_words = time_string.split()

        # outlier
        if time_words[0][-3:] == 'hrs':
            time_words[0] = time_words[0][:2] + ":" + time_words[0][2:4]
        if time_words[0][-1:] == '.':
            time_words[0] = time_words[0][:-1]
        if len(time_words) > 1:
            if time_words[1][-1:] == '.':
                time_words[1] = time_words[1][:-1]
        # rules
        if len(time_words) == 1:
            if re.match(time_pattern, time_words[0]):
                time_string = time_words[0]
            elif re.match(dot_time_pattern, time_words[0]):
                time_string = time_words[0].split(".")[0] + ':' + time_words[0].split(".")[1]
            elif time_words[0] in daytime_to_num:
                time_string = daytime_to_num[time_words[0]]
            elif time_words[0][-2:] == 'pm':
                if len(time_words[0][:-2]) == 1 or len(time_words[0][:-2]) == 2:
                    time_string = str(int(time_words[0][:-2]) + 12) + zz_tail
                elif re.match(time_pattern, time_words[0][:-2]):
                    time_string = str(int(time_words[0][:-2].split(':')[0]) + 12) + ":" + time_words[0][:-2].split(':')[
                        1]
            elif time_words[0][-2:] == 'am':
                if len(time_words[0][:-2]) == 1 or len(time_words[0][:-2]) == 2:
                    time_string = time_words[0][:-2] + zz_tail
                elif re.match(time_pattern, time_words[0][:-2]):
                    time_string = time_words[0][:-2]
            elif time_words[0] == 'dontcare':
                time_string = "9:00"
            elif time_words[0].isdigit() and int(time_words[0]) <= 24:
                time_string = time_words[0] + zz_tail
        elif len(time_words) == 2:
            if time_words[0] == 'after':
                if re.match(time_pattern, time_words[1]):
                    time_string = str(int(time_words[1].split(':')[0]) + 1) + ":" + time_words[1].split(':')[1]
                elif time_words[1][-2:] == 'pm':
                    if len(time_words[1][:-2]) == 1 or len(time_words[1][:-2]) == 2:
                        time_string = str(int(time_words[1][:-2]) + 13) + zz_tail
                elif time_words[1][-2:] == 'am':
                    if len(time_words[1][:-2]) == 1 or len(time_words[1][:-2]) == 2:
                        time_string = str(int(time_words[1][:-2]) + 1) + zz_tail
            elif time_words[0] == 'before':
                if re.match(time_pattern, time_words[1]):
                    time_string = str(int(time_words[1].split(':')[0]) - 1) + ":" + time_words[1].split(':')[1]
                elif time_words[1][-2:] == 'pm':
                    if len(time_words[1][:-2]) == 1 or len(time_words[1][:-2]) == 2:
                        time_string = str(int(time_words[1][:-2]) + 11) + zz_tail
                elif time_words[1][-2:] == 'am':
                    if len(time_words[1][:-2]) == 1 or len(time_words[1][:-2]) == 2:
                        time_string = str(int(time_words[1][:-2]) - 1) + zz_tail
            elif time_words[1] == 'pm':
                if len(time_words[0]) == 1:
                    time_string = str(int(time_words[0]) + 12) + zz_tail
                elif re.match(time_pattern, time_words[0]):
                    time_string = str(int(time_words[0].split(':')[0]) + 12) + ":" + time_words[0].split(':')[1]
            elif time_words[1] == 'am':
                if len(time_words[0]) == 1:
                    time_string = time_words[0] + zz_tail
                elif re.match(time_pattern, time_words[0]):
                    time_string = time_words[0]
            elif time_words[1] == "o'clock":
                if time_words[0].isdigit() and int(time_words[0]) <= 24:
                    time_string = time_words[0] + zz_tail
                elif time_words[0] in word_to_num:
                    time_string = word_to_num[time_words[0]] + zz_tail
        elif len(time_words) == 3:
            if (time_words[0] in word_to_num) and (time_words[1] in word_to_num) and (time_words[2] in word_to_num):
                time_string = word_to_num[time_words[0]] + ":" + str(
                    int(word_to_num[time_words[1]]) + int(word_to_num[time_words[2]]))
        else:
            # default time
            time_string = "12:00"

        converted_time_string = time_string
        return converted_time_string

    def query(self,
              domain: Text,
              constraints: Dict[Text, Text],
              fuzzy_ratio: int = 90):
        """
        Returns the list of entities (dictionaries) for a given domain based on the annotation of the belief state.

        Arguments:
            domain:      Name of the queried domain.
            constraints: Hard constraints to the query results.
        """

        if domain == 'taxi':
            c, t, p = None, None, None

            c = constraints.get('color', [random.choice(self.data[domain]['taxi_colors'])])[0]
            t = constraints.get('type', [random.choice(self.data[domain]['taxi_types'])])[0]
            p = constraints.get('phone', [''.join([str(random.randint(1, 9)) for _ in range(11)])])[0]

            return [{'color': c, 'type': t, 'phone': p}]

        elif domain == 'hospital':

            hospital = {
                'hospital phone': '01223245151',
                'address': 'Hills Rd, Cambridge',
                'postcode': 'CB20QQ',
                'name': 'Addenbrookes'
            }

            departments = [x.strip().lower() for x in constraints.get('department', [])]
            phones = [x.strip().lower() for x in constraints.get('phone', [])]

            if len(departments) == 0 and len(phones) == 0:
                return [dict(hospital)]
            else:
                results = []
                for i in self.data[domain]:
                    if 'department' in self.FUZZY_KEYS[domain]:
                        f = (lambda x: fuzz.partial_ratio(i['department'].lower(), x) > fuzzy_ratio)
                    else:
                        f = (lambda x: i['department'].lower() == x)

                    if any(f(x) for x in departments) and \
                            (len(phones) == 0 or any(i['phone'] == p.strip() for p in phones)):
                        results.append(dict(i))
                        results[-1].update(hospital)

                return results

        else:
            # Hotel database keys:      address, area, name, phone, postcode, pricerange, type, internet, parking,
            # stars, takesbookings (other are ignored) Attraction database keys: address, area, name, phone,
            # postcode, pricerange, type, entrance fee (other are ignored) Restaurant database keys: address, area,
            # name, phone, postcode, pricerange, type, food

            # Train database contains keys: arriveby, departure, day, leaveat, destination, trainid, price, duration
            # The keys arriveby, leaveat expect a time format such as 8:45 for 8:45 am

            results = []
            query = {}

            if domain == 'attraction' and 'entrancefee' in constraints:
                constraints['entrance fee'] = constraints.pop(['entrancefee'])

            for key in self.data_keys[domain]:
                constr_key = str(domain) + "-" + str(key)
                query[key] = constraints.get(constr_key, [])
                if len(query[key]) > 0 and key in ['arriveby', 'leaveat']:
                    if isinstance(query[key][0], str):
                        query[key] = [query[key]]
                    query[key] = [self.time_str_to_minutes(x) for x in query[key]]
                    query[key] = list(set(query[key]))

            for i, item in enumerate(self.data[domain]):
                for k, v in query.items():
                    if len(v) == 0 or item[k] == '?':
                        continue

                    if k == 'arriveby':
                        # TODO: accept item[k] if it is earlier or equal to time in the query
                        # TODO: if the database entry is not ok:
                        # TODO:    break
                        if int(item[k].split(':')[0]) < int(v[0].split(':')[0]):
                            pass
                        elif int(item[k].split(':')[0]) == int(v[0].split(':')[0]) and int(
                                item[k].split(':')[1]) <= int(v[0].split(':')[1]):
                            pass
                        else:
                            break
                    elif k == 'leaveat':
                        # TODO: accept item[k] if it is later or equal to time in the query
                        # TODO: if the database entry is not ok:
                        # TODO:    break
                        if int(item[k].split(':')[0]) > int(v[0].split(':')[0]):
                            pass
                        elif int(item[k].split(':')[0]) == int(v[0].split(':')[0]) and int(
                                item[k].split(':')[1]) >= int(
                                v[0].split(':')[1]):
                            pass
                        else:
                            break
                    elif k in self.FUZZY_KEYS[domain]:
                        ratio = fuzz.partial_ratio(v, item[k])
                        if ratio > fuzzy_ratio:
                            pass
                        else:
                            break
                    elif v == item[k]:
                        pass
                    else:
                        break
                        # TODO: accept item[k] if it matches to the values in query TODO: Consider using fuzzy
                        #  matching! See `partial_ratio` method in the fuzzywuzzy library. TODO: Also, take a look
                        #   into self.FUZZY_KEYS which stores slots suitable for being done in a fuzzy way. TODO: if
                        #    the database entry is not ok: break

                else:  # This gets executed iff the above loop is not terminated
                    result = copy.deepcopy(item)
                    if domain in ['train', 'hotel', 'restaurant']:
                        ref = constraints.get('ref', [])
                        result['ref'] = '{0:08d}'.format(i) if len(ref) == 0 else ref

                    results.append(result)

            if domain == 'attraction':
                for result in results:
                    result['entrancefee'] = result.pop('entrance fee')

            return results
