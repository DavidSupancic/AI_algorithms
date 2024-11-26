import sys
import queue

class Node:
    def __init__(self, state, previous_state, price, heuristic=0):
        self.state = state
        self.previous_state = previous_state
        self.price = price
        self.heuristic = heuristic


def get_distance_data(ss):
    dictOfDistances = {}
    with open(ss, "r") as file:
        row = 0

        for line in file.readlines():
            if line.strip()[0] != "#":

                if row>1: 
                    # dict inside of dict (e.g. {"Umag": {"Buje":13} })
                    list1 = line.split(":")
                    list2 = list1[1].strip().split()
                    dict1 = {}
                    for i in range(len(list2)):
                        list2[i] = list2[i].split(",")
                        dict1[ list2[i][0] ] = list2[i][1]
                    
                    dictOfDistances[list1[0]] = dict1


                elif row == 0:
                    starting_state = line.strip()
                elif row == 1:
                    end_state = line.strip()
                row += 1
    return starting_state, end_state, dictOfDistances

def get_heuristic_data(h):
    dictOfHeuristics = {}
    with open(h, "r") as file:
        for line in file:
            if line.strip()[0] != "#":
                list1 = line.split(":")
                if list1[1]!="":
                    dictOfHeuristics[list1[0]] = int(list1[1].strip())
        
    return dictOfHeuristics


def bfs(ss):
    start_state, end_state, dictOfDistances = get_distance_data(ss)
    end_state = end_state.split(" ")
    open_nodes = queue.Queue()  #[open_state, cost]
    open_nodes.put(Node(start_state, "", 0, 0))
    #best_nodes = {}   #dict {start_state : [path] }
    visited = set()
    found_end_state = False
    path = []

    while open_nodes.qsize()>0:
        current_node = open_nodes.get()
        

        if current_node.state not in end_state:
       
            next_nodes_dict = dictOfDistances[current_node.state]
            next_nodes_keys = list(next_nodes_dict.keys())
            next_nodes_keys.sort()

            for i in range (len(next_nodes_keys)): 
                  
                if next_nodes_keys[i] not in visited:
                    visited.add(next_nodes_keys[i])
                    open_nodes.put(Node(next_nodes_keys[i], current_node, current_node.price+1, 0))

        else: #end_state
            found_end_state = True
            total_cost = current_node.price
            while current_node.previous_state!="":
                path.append(current_node.state)
                current_node = current_node.previous_state
            path.append(start_state)
            path.reverse()
            break

    print("# BFS")
    if found_end_state:
        print("[FOUND_SOLUTION]: yes")
        print("[STATES_VISITED]:", len(visited) )
        print("[PATH_LENGTH]:", len(path))
        print("[TOTAL_COST]: {:.1f}".format(total_cost+1))
        path = " => ".join(path)
        print("[PATH]:", path)

    else:
        print("[FOUND_SOLUTION]: no")
    
    return 


def ucs(ss, start_state, end_state, dictOfDistances, print_result=True):
    open_nodes = [Node(start_state, "", 0, 0)]  #[open_state, cost]
    visited = set()
    found_end_state = False
    path = []
    while len(open_nodes)>0:
        current_node = open_nodes.pop(0)
        
        

        if current_node.state not in end_state:
            next_nodes_dict = dictOfDistances[current_node.state]
            next_nodes_keys = list(next_nodes_dict.keys())

            for i in range (len(next_nodes_keys)): 
                  
                #if next_nodes_keys[i] not in visited:
                visited.add(next_nodes_keys[i])
                dist = next_nodes_dict[next_nodes_keys[i]]
                open_nodes.append(Node(next_nodes_keys[i], current_node, current_node.price+int(dist), 0))

            open_nodes = sorted(open_nodes, key = lambda x: x.price)


        else: #end_state
            found_end_state = True
            total_cost = current_node.price
            while current_node.previous_state!="":
                path.append(current_node.state)
                current_node = current_node.previous_state
            path.append(start_state)
            path.reverse()
            break
    
    if print_result:
        print("# UCS")
        if found_end_state:
            print("[FOUND_SOLUTION]: yes")
            print("[STATES_VISITED]:", len(visited) )
            print("[PATH_LENGTH]:", len(path))
            print("[TOTAL_COST]:", "{:.1f}".format(total_cost))
            path = " => ".join(path)
            print("[PATH]:", path)

        else:
            print("[FOUND_SOLUTION]: no")
    
    return total_cost


def astar(ss, h, start_state, end_state, dictOfDistances, print_result=True):
    dictOfHeuristics = get_heuristic_data(h)
    open_nodes = [Node(start_state, "", 0, 0)]  # h(start_state) does not affect astar
    visited = {start_state:0}
    found_end_state = False
    path = []
    while len(open_nodes)>0:
        current_node = open_nodes.pop(0)    

        if current_node.state not in end_state:
            next_nodes_dict = dictOfDistances[current_node.state]
            next_nodes_keys = list(next_nodes_dict.keys())

            for i in range (len(next_nodes_keys)): 
                
                dist = next_nodes_dict[next_nodes_keys[i]]
                #next_node = Node(next_nodes_keys[i], current_node, 
                                 #current_node.price+int(dist), dictOfHeuristics[next_nodes_keys[i]])
                
                if next_nodes_keys[i] not in visited.keys() or visited[next_nodes_keys[i]] > (current_node.price+int(dist)):
                    visited[next_nodes_keys[i]] = current_node.price+int(dist)
                    
                    open_nodes.append(Node(next_nodes_keys[i], current_node, 
                                        current_node.price+int(dist), dictOfHeuristics[next_nodes_keys[i]]))

            open_nodes = sorted(open_nodes, key = lambda x: (x.price+x.heuristic) )


        else: #end_state
            found_end_state = True
            total_cost = "{:.1f}".format(current_node.price)
            while current_node.previous_state!="":
                path.append(current_node.state)
                current_node = current_node.previous_state
            path.append(start_state)
            path.reverse()
            break
    
    if print_result:
        print("# A-star", h)
        if found_end_state:
            print("[FOUND_SOLUTION]: yes")
            print("[STATES_VISITED]:", len(visited) )
            print("[PATH_LENGTH]:", len(path))
            print("[TOTAL_COST]:", total_cost)
            path = " => ".join(path)
            print("[PATH]:", path)

        else:
            print("[FOUND_SOLUTION]: no")
    
    return total_cost


def check_optimistic(ss, h, end_state, dictOfDistances):
    optimistic = True
    print("# HEURISTIC-OPTIMISTIC", h)

    dictOfHeuristics = get_heuristic_data(h)

    states = list(dictOfHeuristics.keys())
    states.sort()
    print(states)

    for i in range (len(states)):
        real_distance = ucs(ss, states[i], end_state, dictOfDistances, print_result=False) #astar from every state to end_state

        if real_distance < dictOfHeuristics[states[i]]:
            optimistic = False
            print("[CONDITION]: [ERR] h(" + states[i] + ") <= h*: " + "{:.1f}".format(dictOfHeuristics[states[i]])
                  + " <= " + "{:.1f}".format(real_distance))
        else:
            print("[CONDITION]: [OK] h(" + states[i] + ") <= h*: " + "{:.1f}".format(dictOfHeuristics[states[i]])
                  + " <= " + "{:.1f}".format(real_distance))
                

    if (optimistic):
        print("[CONCLUSION]: Heuristic is optimistic.")
    else:
        print("[CONCLUSION]: Heuristic is not optimistic.")
    
    return

def check_consistent(ss, h):
    consistent = True
    print("# HEURISTIC-CONSISTENT", h)

    start_state, end_state, dictOfDistances = get_distance_data(ss)
    dictOfHeuristics = get_heuristic_data(h)

    states = list(dictOfHeuristics.keys())
    states.sort()

    for i in range (len(states)):
        next_states_dict = dictOfDistances[states[i]] 
        next_states_keys = list(next_states_dict.keys())
        next_states_keys.sort()
        H1 = dictOfHeuristics[states[i]]

        for j in range (len(next_states_keys)):
            H2 = dictOfHeuristics[next_states_keys[j]]


            dist_between_states = next_states_dict[next_states_keys[j]]
            if H1 > (H2 + int(dist_between_states)):
                consistent = False
                print("[CONDITION]: [ERR] h("+states[i]+") <= h("+next_states_keys[j]+
                        ") + c: "+"{:.1f}".format(H1)+" <= "+"{:.1f}".format(H2)+" + "+"{:.1f}".format(int(dist_between_states)))
                
            else:
                print("[CONDITION]: [OK] h("+states[i]+") <= h("+next_states_keys[j]+
                        ") + c: "+"{:.1f}".format(H1)+" <= "+"{:.1f}".format(H2)+" + "+"{:.1f}".format(int(dist_between_states)))

    
    if (consistent):
        print("[CONCLUSION]: Heuristic is consistent.")
    else:
        print("[CONCLUSION]: Heuristic is not consistent.")
    
    return

# 

dictOfArgs = {"--alg":False, "--check-optimistic":False, "--check-consistent":False} #dictionary of arguments (e.g. "alg":"astar")

for i in range (1,len(sys.argv)):
    if (sys.argv[i][0:2] == "--"):
        if (sys.argv[i] == "--check-optimistic"):
            dictOfArgs["--check-optimistic"] = True
        elif (sys.argv[i] == "--check-consistent"):
            dictOfArgs["--check-consistent"] = True
        else:
            dictOfArgs[sys.argv[i]] = sys.argv[i+1]

# algorithm
if dictOfArgs["--alg"] == "bfs":
    ss = dictOfArgs["--ss"]
    bfs(ss)

elif dictOfArgs["--alg"] == "ucs":
    ss = dictOfArgs["--ss"]
    start_state, end_state, dictOfDistances = get_distance_data(ss)
    ucs(ss, start_state, end_state, dictOfDistances)

elif dictOfArgs["--alg"] == "astar":
    ss = dictOfArgs["--ss"]
    h = dictOfArgs["--h"]
    start_state, end_state, dictOfDistances = get_distance_data(ss)
    end_state = end_state.split(" ")
    path_len = astar(ss, h, start_state, end_state, dictOfDistances)

elif dictOfArgs["--alg"] == False:
    ss = dictOfArgs["--ss"]
    h = dictOfArgs["--h"]

    if dictOfArgs["--check-optimistic"] == True:
        start_state, end_state, dictOfDistances = get_distance_data(ss)
        end_state = end_state.split(" ")
        check_optimistic(ss, h, end_state, dictOfDistances)

    elif dictOfArgs["--check-consistent"] == True:
        check_consistent(ss,h)

    else:
        print("Wrong input!")

else:
    print("Algorithm does not exist. Use bfs, ucs or astar.")
