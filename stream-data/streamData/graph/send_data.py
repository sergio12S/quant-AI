# from gremlin_python.driver.client import Client
# from gremlin_python.process.anonymous_traversal import traversal
# from gremlin_python.driver.driver_remote_connection import\
#     DriverRemoteConnection
# from datetime import datetime
# import nest_asyncio
# nest_asyncio.apply()


# # server = ''
# server = ''
# port = 8182
# endpoint = 'ws://' + server + ':' + str(port) + '/gremlin'
# transport_args = {'max_content_length': 200000}

# connection = DriverRemoteConnection(endpoint, 'g', **transport_args)
# g = traversal().withRemote(connection)
# client = Client(endpoint, 'g')


# def inject_data(query_string: str, label: str) -> str:
#     return f'g.inject([{query_string}]).unfold().as("m").addV("{label}").as("v").select("m").unfold().as("kv").select("v").property(select("kv").by(keys), select("kv").by(values)).id().toList()'


# def build_request(message: dict, timestamp: datetime) -> dict:
#     query_string = '['
#     for k, v in message.items():
#         query_string += f"'{k}': '{v}',"
#     query_string += f"'year': {timestamp.year},"
#     query_string += f"'month': {timestamp.month},"
#     query_string += f"'weekday': {timestamp.weekday()},"
#     query_string += f"'day': {timestamp.day},"
#     query_string += f"'hour': {timestamp.hour},"
#     query_string += f"'minute': {timestamp.minute},"
#     query_string += f"'second': {timestamp.second}"
#     query_string += ']'
#     return query_string


# def processing_data(df):
#     list_maps = []
#     timestamp = datetime.now()
#     for i in df:
#         query_string = build_request(message=i, timestamp=timestamp)
#         list_maps.append(query_string)
#     return list_maps


# def create_query_string(df):
#     list_maps = processing_data(df)
#     query_string_full = ''
#     for i in range(len(list_maps)):
#         query_string = inject_data(list_maps[i], df[i].get('name'))
#         if query_string_full == '':
#             query_string_full = query_string
#         else:
#             query_string_full = query_string_full + ";" + query_string
#     return query_string_full


# def send_data_to_gremlin(df):
#     query_string_full = create_query_string(df)
#     if query_string_full:
#         req = client.submit(query_string_full).all().result()
#         if len(req):
#             return '200'
