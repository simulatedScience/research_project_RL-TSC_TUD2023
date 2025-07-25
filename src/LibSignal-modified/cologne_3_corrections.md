# Problem description:
Looking at the simulation results, I noticed that some vehicles spawn in the middle of the network, which leads to problematic traffic situations. For example, vehicle `140163_414` spawns on a right-turning lane but wants to turn left there. Simoultaneously, another vehicle arrives on the left turning lane, but wants to turn right. This leads to a deadlock situation, where both vehicles are waiting for the other to move. This stalls the simulation and leads to long wait times that are not representative of any TSC-agent's performance.

To fix this, I implement a tool to modify the routes of all vehicles that spawn or despawn in the middle of the network. To do this, I first filter for all such violating vehicles, then find route extensions that complete the current route to a valid one.
For this, we need to know the incoming and outgoing edges of the network such that we can filter the .rou.xml file.

## Implementation:
1. Given incoming and outgoing edges, create a set of all internal edges.
2. Split the .rou.xml file into parts: 
    - Vehicles that spawn on internal or outgoing edges (collect these in an `invalid_spawn_set`)
    - Vehicles that despawn on internal or incoming edges (collect these in an `invalid_despawn_set`)
    - Vehicles that spawn on incoming edges (collect these in an `valid_spawn_set`)
    - Vehicles that despawn on outgoing edges (collect these in an `valid_despawn_set`)
3. For each edge in `invalid_spawn_set` and `invalid_despawn_set`, make a list of valid route extensions using `valid_spawn_set` and `valid_despawn_set`.
4. For each vehicle with an invalid spawn, sample a random route extension from the list of valid route extensions and append it to the vehicle's route.
5. For each vehicle with an invalid despawn, sample a random route extension from the list of valid route extensions and prepend it to the vehicle's route.

### Incoming lanes:
31864804
31864804
241660957#0
241660957#0
-31800061#0
-41910185#2
4045330
-130160207#0
-241660955#17
-241660955#17
41910184
4045329#5
4999334
41910186
319261593#12
319261593#12
-5229966#3

### Outgoing lanes:
-31864804
-31864804
4145590#0
4145590#0
31800061#0
4045332#0
41910185#0
-4045330
4145589#0
130160207#0
241660955#17
241660955#17
-41910184
-4045329#5
-4999334
-41910186
4999331#0
4999331#0