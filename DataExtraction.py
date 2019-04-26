import pandas as pd
import os
from sqlalchemy import create_engine
import sqlite3
import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import itertools

from rdp import rdp


def angle(directions):
    """Return the angle between vectors
    """
    vec2 = directions[1:]
    vec1 = directions[:-1]

    norm1 = np.sqrt((vec1 ** 2).sum(axis=1))
    norm2 = np.sqrt((vec2 ** 2).sum(axis=1))
    cos = (vec1 * vec2).sum(axis=1) / (norm1 * norm2)
    return np.arccos(cos)


def df2sqlite_v2(dataframe, db_name):
    disk_engine = create_engine('sqlite:///' + db_name + '.db')
    dataframe.to_sql(db_name, disk_engine, if_exists='replace', chunksize=1000)

    """Bundan onceki !!!! Bunu unutma updated_stats V3 icin bunu yapmak daha dogru olabilir. Dont know the difference
    #     dataframe.to_sql(db_name, disk_engine ,if_exists='append')"""


def convert_csvgames_to_sqlite(gameIds):
    for gameId in gameIds:
        file_string = "data/tracking_gameId_" + str(gameId) + ".csv"
        print(file_string)
        df2sqlite_v2(pd.read_csv(file_string), str(gameId))


def personnel_abbrevation(data):
    for i in data.index:
        off_personnel = data.at[i, "personnel.offense"]
        positions = off_personnel.split(',')
        rb_and_te = (positions[:2])
        assert len(rb_and_te) == 2
        number_of_rb = rb_and_te[0].strip().split(' ')[0]
        number_of_te = rb_and_te[1].strip().split(' ')[0]

        data.at[i, "personnel.offense"] = number_of_rb + number_of_te

    print("ali")
    print("stop")
    return data


class NFLRouteVisualizer(object):

    def __init__(self, plays_database):
        conn = sqlite3.connect(plays_database)
        start_time = time.time()

        self.pass_plays = pd.read_sql_query("SELECT * FROM plays WHERE PassResult IS NOT NULL", conn)
        self.run_plays = pd.read_sql_query("SELECT * FROM plays WHERE PassResult IS NULL", conn)

        conn_players = sqlite3.connect("players.db")
        conn_games = sqlite3.connect("games.db")
        self.games = pd.read_sql_query('SELECT * FROM games', conn_games)
        self.players = pd.read_sql_query('SELECT * FROM players', conn_players)

        self.gameIds = list(self.games.gameId.unique())

        conn_game1 = sqlite3.connect("2017090700.db")

        self.game1 = pd.read_sql_query("SELECT * FROM '2017090700'", conn_game1)
        self.gameId = '2017090700'
        print(len(self.game1))
        self.game_pass_plays_ids = []

        self.player_trajectory_dict = {}

        #  convert_csvgames_to_sqlite(gameIds)  # one time only

    def pass_plays_in_a_game(self):
        current_game_id = list(self.game1.gameId)[0]

        pass_plays_in_game = self.pass_plays.loc[self.pass_plays.gameId == current_game_id]

        self.game_pass_plays_ids = list(pass_plays_in_game.playId)
        pass_play_tracking_data = self.game1.loc[self.game1['playId'].isin(self.game_pass_plays_ids)]

        game1_pass_plays_tracking_data = pd.merge(pass_play_tracking_data, pass_plays_in_game, on=['playId'])
        game1_pass_plays_tracking_data = pd.merge(game1_pass_plays_tracking_data,
                                                  self.players[['nflId', 'PositionAbbr']], on='nflId', how='left')

        print("total game tracking data row number is {}".format(len(game1_pass_plays_tracking_data)))
        print(game1_pass_plays_tracking_data.columns)

        return game1_pass_plays_tracking_data

    def visualize_all_passing_plays(self, data):
        counter = 0
        print(self.game_pass_plays_ids)
        for play_id in [724, 2317, 2756, 4251]:
            print("Play id is {}".format(play_id))
            receiver_data = self.get_receivers_in_a_play(data, play_id)

            if receiver_data['PassResult'].unique()[0] == 'S':
                continue
            if 'pass_forward' not in receiver_data['event'].unique():
                continue

            opposing_direction = self.trajectory_calculator(receiver_data, play_id)
            counter = counter + 1
            print("Play Counter is {}".format(counter))
            self.turning_point_identifier(play_id, opposing_direction)

    def get_receivers_in_a_play(self, data, pass_play_Id):
        receivers = ['WR', 'TE', 'RB']
        # go over each pass play and identify the receivers, running backs and tight ends

        current_play = data.loc[data['playId'] == pass_play_Id]

        print("data points for current play : {}".format(len(current_play)))
        receivers_of_current_play = current_play.loc[
            np.logical_or(current_play['PositionAbbr'].isin(receivers), current_play['team'] == 'ball')].reset_index(
            drop=True)

        # receivers_of_current_play.to_csv('receivers.csv')
        print("data points for receivers in current play : {}".format(len(receivers_of_current_play)))

        return receivers_of_current_play

    def trajectory_calculator(self, data, play_id):
        player_list = list(data['displayName'].unique())

        single_player = data.iloc[0]
        football = data.loc[data['displayName'] == 'football']

        player_starting_x = single_player[['x']].unique()[0]
        ball_starting_x = football[['x']].iloc[0].unique()[0]
        opposing_direction = False

        if ball_starting_x > player_starting_x:
            opposing_direction = True

        for player in player_list:

            player_play_data = data.loc[data['displayName'] == player]

            player_xy_data = player_play_data[['x', 'y', 'dir', 'event', 'frame.id']].reset_index(drop=True)

            # General Play information
            play_data = player_play_data[
                ['down', 'yardsToGo', 'possessionTeam', 'yardlineSide', 'yardlineNumber', 'personnel.offense',
                 'PassResult', 'playDescription', 'team', 'PositionAbbr']].iloc[0]
            play_data['receiver_set'] = ""

            if player != 'football':

                # start tracking data from the ball snap frame

                event_values = player_xy_data['event'].unique()
                ballsnap_frame_index = int(player_xy_data.index[player_xy_data['event'] == 'ball_snap'][0])

                if 'pass_shovel' in event_values:

                    pass_arrived_frame_index = int(player_xy_data.index[player_xy_data['event'] == 'pass_shovel'][0])

                else:
                    pass_arrived_frame_index = int(
                        player_xy_data.index[player_xy_data['event'] == 'pass_forward'][0]) + 20
                    frame_id = player_xy_data[player_xy_data['event'] == 'pass_forward']
                # print(int(frame_id['frame.id']))

                # get the adjusted frame that pass was thrown
                pass_forward_adjusted_frame_index = pass_arrived_frame_index - 10 - ballsnap_frame_index

                # Start trajectory data from ball snap
                player_xy_data_post_snap = player_xy_data.iloc[ballsnap_frame_index:pass_arrived_frame_index]

                if opposing_direction:

                    updated_x = np.negative(player_xy_data_post_snap['x'])
                    updated_y = np.negative(player_xy_data_post_snap['y'])
                    trajectory = np.column_stack((updated_x, player_xy_data_post_snap['y']))
                    self.player_trajectory_dict[player, play_id] = [trajectory,
                                                                    updated_x,
                                                                    player_xy_data_post_snap['y'],
                                                                    np.asarray(player_xy_data_post_snap['dir']),
                                                                    pass_forward_adjusted_frame_index, play_data]
                else:

                    trajectory = np.column_stack((player_xy_data_post_snap['x'], player_xy_data_post_snap['y']))
                    self.player_trajectory_dict[player, play_id] = [trajectory,
                                                                    np.asarray(player_xy_data_post_snap['x']),
                                                                    np.asarray(player_xy_data_post_snap['y']),
                                                                    np.asarray(player_xy_data_post_snap['dir']),
                                                                    pass_forward_adjusted_frame_index, play_data]

            else:
                if opposing_direction:
                    updated_x = np.negative(player_xy_data['x'])

                    trajectory = np.column_stack((updated_x, player_xy_data['y']))
                    self.player_trajectory_dict[tuple([player, play_id])] = [trajectory,
                                                                             updated_x,
                                                                             player_xy_data['y'],
                                                                             np.asarray(player_xy_data['dir']), 0,

                                                                             play_data]
                else:

                    trajectory = np.column_stack((player_xy_data['x'], player_xy_data['y']))
                    self.player_trajectory_dict[tuple([player, play_id])] = [trajectory,
                                                                             np.asarray(player_xy_data['x']),
                                                                             np.asarray(player_xy_data['y']),
                                                                             np.asarray(player_xy_data['dir']), 0,
                                                                             play_data]

        return opposing_direction

    def turning_point_identifier(self, play_id, opposing_direction):
        print("ali")
        print("stop")
        fig = plt.figure()
        color_array = ["grey", "orange", "green", "blue", 'brown', 'black', 'yellow']

        ball = self.player_trajectory_dict[tuple(['football', play_id])]
        possible_ball_points = np.asarray([23.36, 26.66335, 29.9667])
        ball_starting_x = ball[1][0]
        ball_starting_y = ball[2][0]
        ball_point = possible_ball_points[(np.abs(possible_ball_points - ball_starting_y)).argmin()]
        ball_starting_np = np.column_stack((ball_starting_x, ball_starting_y))
        right_side_receivers = []
        left_side_receivers = []
        # the first for loop takes the ball as the origin on the graph and recalculates trajectories of players
        for i, (k, v) in enumerate(self.player_trajectory_dict.items()):

            player_name, p_id = k
            trajectory, x, y, direction, pass_thrown_frame, play_data = v
            if player_name != 'football':

                new_x = [ball_starting_x - i for i in x]
                if opposing_direction:
                    new_y = [(ball_starting_y - i) for i in y]
                else:
                    new_y = [-(ball_starting_y - i) for i in y]

                new_trajectory = np.column_stack((new_x, new_y))
                self.player_trajectory_dict[k] = [new_trajectory, new_x, new_y, direction, pass_thrown_frame, play_data]

                # Add Running Back to both sides
                if play_data[['PositionAbbr']].unique()[0] == 'RB':
                    left_side_receivers.append(
                        [player_name, p_id, new_trajectory, new_x, new_y, direction, pass_thrown_frame, play_data])
                    right_side_receivers.append(
                        [player_name, p_id, new_trajectory, new_x, new_y, direction, pass_thrown_frame, play_data])
                else:
                    if new_y[0] > 0:
                        left_side_receivers.append(
                            [player_name, p_id, new_trajectory, new_x, new_y, direction, pass_thrown_frame, play_data])
                    else:
                        right_side_receivers.append(
                            [player_name, p_id, new_trajectory, new_x, new_y, direction, pass_thrown_frame, play_data])

            else:
                self.player_trajectory_dict[k] = [[0], [0], np.asarray([0, 0]), 0, 0, play_data]
        # first_route = right_side_receivers[0][2]
        # second_route = right_side_receivers[1][2]
        # first_route = tuple(map(tuple, first_route))
        # second_route = tuple(map(tuple, second_route))
        # print(type(first_route))
        # print(type(list(second_route)))

        # route1[2] [route1[2][:, 0] > 0 & (route1[2[:, 0] > -5)]

        right_side_route_intersection_matrix = [
            [route1[0], route2[0],
             Polygon(route1[2][(5 > route1[2][:, 0]) & (route1[2][:, 0] > 0)]).intersects(
                 Polygon(route2[2][(5 > route2[2][:, 0]) & (route2[2][:, 0] > 0)]))] for
            (route1, route2) in itertools.combinations(right_side_receivers, 2) if
            len(route1[2][(5 > route1[2][:, 0]) & (route1[2][:, 0] > 0)]) > 2 and len(
                route2[2][(5 > route2[2][:, 0]) & (route2[2][:, 0] > 0)]) > 2]

        left_side_route_intersection_matrix = [
            [route1[0], route2[0],
             Polygon(route1[2][(5 > route1[2][:, 0]) & (route1[2][:, 0] > 0)]).intersects(
                 Polygon(route2[2][(5 > route2[2][:, 0]) & (route2[2][:, 0] > 0)]))] for
            (route1, route2) in itertools.combinations(left_side_receivers, 2) if
            len(route1[2][(5 > route1[2][:, 0]) & (route1[2][:, 0] > 0)]) > 2 and len(
                route2[2][(5 > route2[2][:, 0]) & (route2[2][:, 0] > 0)]) > 2]
        """ for (route1, route2) in itertools.combinations(left_side_receivers, 2):
            print(len(route1[2][route1[2][:, 0] > 0]))
            print(route1[2][route1[2][:, 0] > 0])
            print(len(route2[2][route2[2][:, 0] > 0]))
            print(route2[2][route2[2][:, 0] > 0])"""

        possible_route_intersection_on_left = [matrixpoint[2] for matrixpoint in left_side_route_intersection_matrix]
        possible_route_intersection_on_right = [matrixpoint[2] for matrixpoint in right_side_route_intersection_matrix]

        # If there are no intersecting routes on the play, stop the method execution
        # if True not in possible_route_intersection_on_left and True not in possible_route_intersection_on_right:
        #     self.player_trajectory_dict.clear() # Clear all the previous plays
        #    return

        plt.axvline(x=0, linestyle='dashed', label="Line of Scrimmage")
        for i, (k, v) in enumerate(self.player_trajectory_dict.items()):
            player_name, play_id = list(k)
            print(k)
            # print("player name: {}".format(player_name))

            ax = fig.add_subplot(111)

            trajectory, x, y, direction, pass_thrown_frame, play_data = v

            if player_name == 'football':
                ax.plot(x[0], y[0], color='black', marker="8", label='Football', markersize=8)

            else:
                # print(trajectory.shape)
                simplified_trajectory = rdp(trajectory, epsilon=1)
                sx, sy = simplified_trajectory.T
                # Visualize trajectory and its simplified version.

                # Define a minimum angle to treat change in direction
                # as significant (valuable turning point).
                min_angle = np.pi / 40.0  # np.pi = 180 degrees

                # Compute the direction vectors on the simplified_trajectory.
                directions = np.diff(simplified_trajectory, axis=0)
                theta = angle(directions)

                # Select the index of the points with the greatest theta.
                # Large theta is associated with greatest change in direction.
                idx = np.where(theta > min_angle)[0] + 1

                # Visualize valuable turning points on the simplified trajectory.

                ax.plot(x[0], y[0], color='black', marker="x")
                # ax.plot(plt.axes().axhline(linewidth=4, color='r'))
                ax.plot(x, y, color=color_array[i], label=k)  # str(k) + ' ' + self.gameId)
                # ax.plot(sx, sy, color=color_array[i], label=k)
                tpointsx = (sx[idx])
                tpointsy = (sy[idx])

                if i == 0:
                    ax.plot(sx[idx], sy[idx], 'ro', markersize=4, label='Turning points')
                    ax.plot(np.asarray(x)[pass_thrown_frame], np.asarray(y)[pass_thrown_frame], color='black',
                            marker="X", label='pass thrown')

                else:
                    # ax.text(15, -25, play_data['playDescription'], fontsize=6)
                    print(play_data['playDescription'])
                    ax.plot(sx[idx], sy[idx], 'ro', markersize=4)
                    ax.plot(np.asarray(x)[pass_thrown_frame], np.asarray(y)[pass_thrown_frame], color='black',
                            marker="X")

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_ylim(bottom=ball_starting_y, top=0 - (53.3 - ball_starting_y))
            # ax.axhline(y= 0,xmin = -1.5,xmax = 1.5,linestyle='dashed')
            # ax.margins(x=0)

            ax.legend(loc='best', prop={'size': 5})

        plt.plot((-0.5, 0.5), (0, 0), linestyle='dashed', color='black', label='hash left')

        plt.show()
        fig.savefig(str(play_id) + '.png')
        # plt.savefig(str(play_id) + '.png', bbox_inches='tight')

        self.player_trajectory_dict.clear()


DE = NFLRouteVisualizer("plays.db")

pass_plays_data = DE.pass_plays_in_a_game()  # all pass plays in a game
pass_plays_data = personnel_abbrevation(pass_plays_data)  # adds position abbrevation to the data

# For each play extract receiver tracking data
DE.visualize_all_passing_plays(pass_plays_data)
# receivers_tracking_data = DE.get_receivers_in_a_play(pass_plays_data, 2756)  # give play as an example

# For each receiver, create their trajectory for that play from ball snap to pass caught frames
# DE.trajectory_calculator(receivers_tracking_data)

# Identify the turning points in their routes
# DE.turning_point_identifier()
