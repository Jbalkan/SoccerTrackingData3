"""

Tracking_Fatigue.py

written by Jeff Balkanski

"""


def estimate_player_energy_expenditure(team1_players, team0_players, match, metric='VeBDA'):
    """ estaimte player energy expenditure

    Keyword Arguments:
        metric {str} -- metric of energy expenditure (default: {'VeBDA'})
    """
    # get for all players
    all_players = list(team0_players.items()) + list(team1_players.items())
    for (num, player) in all_players:
        if metric == 'VeBDA':
            VeBDA = [sum(player.a_magnitude[:i]) for i in range(len(player.a_magnitude))]
            player.VeBDA = VeBDA

        elif metric == 'metabolic':
            print('not implemented')
            break
