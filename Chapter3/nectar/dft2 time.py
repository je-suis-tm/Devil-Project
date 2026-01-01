# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:55:19 2025

@author: Administrator
"""

#!/usr/bin/env python
# coding: utf-8

# each time step, an agent will move to the nearest resource unless the nearest resource is outside of its current search radius or occupied by too many agents. Then the agent will search for the second nearest until all options are exhausted. If no eligible resource is available for the agent, the agent stays put. in a reproduction timestep, the female agents search for the resource and the male agents search for the females. The male agent with the highest fitness has higher priority to search for female agents. It always selects the female agents with highest fitness within its search radius, similar to resource searching.if a male could not find an eligible female, it will stay put and lose one level of current fitness as it gains no access to resource and get injured from fighting for female. There is no current fitness gain for male during mating season.

# each agent has an attribute called fitness, current and max fitness are randomly assigned at birth, fitness mechanism includes:
# 
# * higher current fitness leads to lower chance of natural mortality
# * higher current fitness leads to higher number of offsprings
# * higher current fitness leads to larger resource searching radius per time step
# * higher current fitness leads to higher priority to look for resource
# * higher current fitness leads to higher chance of resource monopoly once fitness is above a threshold
# * each time step an agent does not have resource, current fitness minus one
# * each time step an agent has resource, current fitness plus one unless max fitness is reached
# * when current fitness drops below zero, the agent dies

# the map is a 10 by 10 lattice with roads and habitats. there is a chance of roadkill when agent needs to cross the road. resources are more likely to spawn along the roads.

# ### packages

# In[1]:


import numpy as np
import ast
from collections import deque
import pickle
import pandas as pd


# ### functions

# In[2]:


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def sort_coordinates_by_distance(coordinates, reference_point, distance_matrix, all_coordinates):
    """Sort a list of coordinates based on their distance from a reference point using the precomputed distance matrix."""
    
    # Find the index of the reference_point in all_coordinates
    reference_index = all_coordinates.index(reference_point)
    
    # Sort the coordinates based on precomputed distances
    sorted_coordinates = sorted(coordinates, key=lambda point: distance_matrix[reference_index, all_coordinates.index(point)], reverse=True)
    
    return sorted_coordinates


# In[3]:


def get_neighbors(x, y):
    """
    Returns the 8 neighboring coordinates of a cell (x, y) on a grid.

    Neighbors include:
    Top-left, Top, Top-right
    Left,        Right
    Bottom-left, Bottom, Bottom-right
    """
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    
    neighbors = [(x + dx, y + dy) for dx, dy in directions]
    return neighbors


# In[4]:


def shortest_path(x1, y1, x2, y2):
    start = (x1, y1)
    goal = (x2, y2)

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    queue = deque([start])
    visited = set([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    # Reconstruct the path
    if goal not in parent:
        return []  # No path found

    path = []
    curr = goal
    while curr:
        path.append(curr)
        curr = parent[curr]
    path.reverse()

    return path


# In[5]:


#initialize all variables needed for simulation
def initialize_agents(num_of_steps, num_of_agents, all_coordinates, min_fitness, max_fitness, max_age, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
        
    #random agent loc
    agent_loc=[all_coordinates[ac] for ac in np.random.choice(len(all_coordinates), num_of_agents)]
    agent_home=[ac for ac in agent_loc]

    #agent has max fitness and current fitness
    #current fitness would minus one without resource at each time step
    #current fitness would add one with resource at each time step as long as max fitness is not reached
    agent_max_fitness=np.random.randint(low=min_fitness,high=max_fitness,size=num_of_agents).tolist()
    agent_current_fitness=[np.random.randint(low=min_fitness-1,high=amf) for amf in agent_max_fitness]

    #tracking agent and resource location
    agent_mat=[[al for al in agent_loc],]
    resource_mat=[]

    #tracking living agents
    alive_agents=[agent_ind for agent_ind in range(num_of_agents)]
    
    #assign age to each agent
    agent_age=np.random.randint(low=0,high=max_age,size=num_of_agents).tolist()

    #assign gender to each agent
    agent_gender=np.random.choice([-1,1], num_of_agents).tolist()

    #summary stats
    metrics=['roadkill', 'starvation', 'monopoly', 'natural mortality','newborn',
             'mating','total alive','immigration','emigration','dft1 susceptible','dft1 exposed','dft1 infected',
             'dft2 susceptible','dft2 exposed','dft2 infected','average fitness','dft1 boundary','dft2 boundary',
             'coinfection']
    vals=[[0 for _ in range(num_of_steps)] for _ in range(len(metrics))]
    stats=dict(zip(metrics,vals))
    stats['dft1 boundary']=[None]*num_of_steps
    stats['dft2 boundary']=[None]*num_of_steps
    
    #diease stats
    dft1_prognosis={}
    dft1_latency_mat={}
    dft1_exposed_mat=[[],]
    dft1_infected_mat=[[],]    
    dft1_map_mat=[]
    
    dft2_prognosis={}
    dft2_latency_mat={}
    dft2_exposed_mat=[[],]
    dft2_infected_mat=[[],]    
    dft2_map_mat=[]

    return agent_home, agent_loc, agent_max_fitness, agent_current_fitness, agent_mat, resource_mat, agent_age, agent_gender, alive_agents, stats, dft1_prognosis, dft1_exposed_mat, dft1_infected_mat, dft1_latency_mat, dft1_map_mat, dft2_prognosis, dft2_exposed_mat, dft2_infected_mat, dft2_latency_mat, dft2_map_mat


# In[6]:


#each time step each agent gets older 
#natural mortality occurs in this function
def apply_natural_aging(agent_age, agent_current_fitness, alive_agents, agent_loc, age_prob,
                        age_denom, max_fitness, timestep, stats, dft1_prognosis, dft2_prognosis, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
       
    #natural aging
    aleatoire=np.random.random(size=len(agent_loc))
    for agent_ind in [agent_id for agent_id in alive_agents]:
            
        #higher fitness lead to lower mortality rate
        #if random rate inflates above one with adj
        #reset to 1
        fitness_adj=1+(1-agent_current_fitness[agent_ind]/max_fitness)/10
        random_rate=fitness_adj*aleatoire[agent_ind]
        if random_rate>1:
            random_rate=1
            
        #natural mortality
        if random_rate<=age_prob[int(agent_age[agent_ind]//age_denom)]:
            agent_loc[agent_ind]=(-1,-1)
            stats['natural mortality'][timestep]+=1
            alive_agents.remove(agent_ind)
            if agent_ind in dft1_prognosis:
                del dft1_prognosis[agent_ind]
            if agent_ind in dft2_prognosis:
                del dft2_prognosis[agent_ind]
                    
        #natural aging
        else:
            agent_age[agent_ind]+=1
            
    return agent_age, agent_loc, alive_agents, stats, dft1_prognosis, dft2_prognosis


# In[7]:


#update agent inflow and outflow
def inflow_outflow(agent_home, alive_agents, agent_loc, agent_age, agent_gender, agent_max_fitness, agent_current_fitness,
                   border_row, border_area, beyond, p_inflow, p_outflow, timestep, max_age, max_fitness, min_fitness,
                          stats, dft1_intro, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, inflow_dft1_infected_rate,
                   min_dft1_latency_period, max_dft1_latency_period, age_denom, dft1_latency_mat, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
    
    #find agents at the border area
    border_agent=[agent_ind for agent_ind in alive_agents if agent_loc[agent_ind][0]<border_row]
    
    #random number matrices
    aleatoire1=np.random.random(size=beyond)
    aleatoire2=np.random.random(size=len(border_agent))
    
    #remove agents moving out of the map
    for b_ind in np.where(aleatoire2<p_outflow)[0]:
        agent_loc[border_agent[b_ind]]=(-1,-1)
        stats['emigration'][timestep]+=1
        alive_agents.remove(border_agent[b_ind])
        if border_agent[b_ind] in dft1_prognosis:
            del dft1_prognosis[border_agent[b_ind]]
        
    #add agents moving into the map
    inflow_num=aleatoire1[aleatoire1<p_inflow].shape[0]
    stats['immigration'][timestep]+=inflow_num
        
    # Append new agents
    alive_agents+=[agent_ind for agent_ind in range(len(agent_loc),len(agent_loc)+inflow_num)]
    extra_loc=[border_area[b_ind] for b_ind in np.random.choice(range(len(border_area)),size=inflow_num)]
    agent_loc+=extra_loc
    agent_home+=extra_loc
    agent_age+=np.random.randint(low=age_denom,high=max_age,size=inflow_num).tolist()
    agent_gender+=np.random.choice([-1,1],inflow_num).tolist()
    new_max_fitness=np.random.randint(low=min_fitness,high=max_fitness,size=inflow_num).tolist()
    new_current_fitness=[np.random.randint(low=min_fitness-1,high=nmf) for nmf in new_max_fitness]
    agent_max_fitness+=new_max_fitness
    agent_current_fitness+=new_current_fitness
    
    #disease part
    dft1_exposed=[agent_ind for agent_ind in dft1_exposed_mat[-1]]
    dft1_infected=[agent_ind for agent_ind in dft1_infected_mat[-1]]
    if timestep==dft1_intro:
        
        #introduce patient zero and randomly assign its infected date based on latency period range
        dft1_infected.append(alive_agents[-1])
        dft1_prognosis[alive_agents[-1]]=[]
        dft1_prognosis[alive_agents[-1]].append(timestep-np.random.randint(min_dft1_latency_period, max_dft1_latency_period))
        
    #after patient zero,immigration could be any class of sei
    if timestep>dft1_intro:
        
        #select fixed numbers of immigration candidates
        #turn half into exposed half into infected
        candidates=np.random.choice(alive_agents[-inflow_num:],
                 size=int(inflow_num*inflow_dft1_infected_rate),replace=False).tolist()
        candidates_e=candidates[:int(len(candidates)/2)]
        candidates_i=candidates[int(len(candidates)/2):]
        dft1_exposed+=candidates_e
        dft1_infected+=candidates_i
        
        #randomly assign exposed date based on latency period range for both class e and i
        aleatoire=np.random.randint(low=min_dft1_latency_period,high=max_dft1_latency_period-1,size=len(candidates)).astype(np.float64)
        
        #higher fitness leads to longer latency period
        adj_factors=[agent_current_fitness[agent_ind]/max_fitness for agent_ind in candidates]
        aleatoire*=adj_factors
        aleatoire=np.floor(aleatoire).astype(np.int64)
        aleatoire[aleatoire<min_dft1_latency_period]=min_dft1_latency_period
        aleatoire_l=[np.random.randint(rd_val,max_dft1_latency_period) for rd_val in aleatoire[:len(candidates_e)]]
        for c_ind,agent_ind in enumerate(candidates):
            dft1_prognosis[agent_ind]=[]
            dft1_prognosis[agent_ind].append(timestep-aleatoire[c_ind])
        for c_ind,agent_ind in enumerate(candidates_e):
            dft1_latency_mat[agent_ind]=aleatoire_l[c_ind]
            
    #update stats
    dft1_exposed_mat.append(list(set(dft1_exposed).intersection(set(alive_agents))))
    dft1_infected_mat.append(list(set(dft1_infected).intersection(set(alive_agents))))
    
    return agent_home, agent_loc, agent_age, agent_gender, agent_max_fitness, agent_current_fitness, alive_agents, stats, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_latency_mat


# In[8]:


# Generate random resource coordinates and capacities
def generate_resources(all_coordinates, prob_arr, min_resource, max_resource, max_resource_cap, resource_mat, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
    
    # Generate random resource coordinates
    num_of_resources=np.random.randint(low=min_resource,high=max_resource)
    resource_loc=np.random.choice(len(all_coordinates),size=num_of_resources,replace=False,p=prob_arr).tolist()
        
    # Generate random resource max capacity
    resource_num=dict(zip(resource_loc,np.random.randint(low=1,high=max_resource_cap,size=num_of_resources)))
        
    # Generate resource act capacity
    resource_count=dict(zip(resource_loc,[[] for _ in range(len(resource_loc))]))
    #resource_mat.append([rl for rl in resource_loc])
        
    return resource_loc, resource_num, resource_count


# In[9]:


#update agent location and behavior associated with resource
def update_agents_status(agent_home, alive_agents, agent_loc, agent_current_fitness, agent_max_fitness, road, 
                          roadkill_prob, monopoly_prob, fitness_threshold, distance_matrix, 
                          all_coordinates, all_shortest_paths, resource_count, resource_num, 
                          stats, timestep, dft1_prognosis, dft2_prognosis, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
    
    #random number matrices
    aleatoire1=np.random.random(size=len(alive_agents))
    aleatoire2=np.random.random(size=len(alive_agents))
    
    for ind,agent_ind in enumerate(alive_agents):
    
        #randomly find destination within its fitness level
        #higher prob where resources are present
        distance_arr=np.array(distance_matrix)[all_coordinates.index(agent_home[agent_ind]),]
        candidates=np.where(distance_arr<=agent_current_fitness[agent_ind])[0]
        prob_arr=np.where(np.isin(candidates, list(resource_count.keys())), 0.4, 0.2)
        destination=np.random.choice(candidates,p=prob_arr/prob_arr.sum())
    
        #skip the starting point
        route=all_shortest_paths[all_coordinates.index(agent_loc[agent_ind])][destination][1:]
    
        #if crossing the road
        #certain probability to become roadkill
        if len(set(route).intersection(set(road)))!=0 and aleatoire1[ind]<roadkill_prob:
            agent_loc[agent_ind]=(-1,-1)
            stats['roadkill'][timestep]+=1
            alive_agents.remove(agent_ind)
            if agent_ind in dft1_prognosis:
                del dft1_prognosis[agent_ind]
            if agent_ind in dft2_prognosis:
                del dft2_prognosis[agent_ind]
    
        #if not crossing the road or crossing safely
        #assigned to the position
        else:    
            agent_loc[agent_ind]=all_coordinates[destination]
    
            #if assigned to resource
            #resource occupancy plus one
            if destination in resource_count:
                resource_count[destination].append(agent_ind)
                
                #high current fitness can create resource monopoly based on prob
                if agent_current_fitness[agent_ind]>=fitness_threshold and aleatoire2[ind]<monopoly_prob:
                    resource_num[destination]=1
                    stats['monopoly'][timestep]+=1
    
            #if not, fitness minus one
            else:
                agent_current_fitness[agent_ind]-=1
    
                #agent will die if fitness is negative
                if agent_current_fitness[agent_ind]<0 and agent_ind in alive_agents:
                    agent_loc[agent_ind]=(-1,-1)
                    stats['starvation'][timestep]+=1
                    alive_agents.remove(agent_ind)
                    if agent_ind in dft1_prognosis:
                        dft1_prognosis[agent_ind].append(timestep)
                    if agent_ind in dft2_prognosis:
                        dft2_prognosis[agent_ind].append(timestep)
    
    #calculate current and max cap of each resource
    #filter where resource_current_cap is over resource_max_cap
    resource_current_cap=np.array([len(rc) for rc in resource_count.values()])
    resource_max_cap=np.array(list(resource_num.values()))
    resource_over_cap=np.array(list(resource_count.keys()))[np.where(resource_current_cap>resource_max_cap)[0]]
    no_resource=[]
    
    #if resource_current_cap is over resource_max_cap
    #sort agents by fitness level
    #only the top n gets consumption
    #n is determined by resource_max_cap
    for roc in resource_over_cap:
        fitness_order=sorted(resource_count[roc],key=lambda x:agent_current_fitness[x],reverse=True)
        no_resource+=fitness_order[resource_num[roc]:]
    
    #adjust fitness level of agents which get to consume resources
    allocated=[agent_ind for rc in resource_count.values() for agent_ind in rc]
    for agent_ind in allocated:
        if agent_ind not in no_resource:
            
            #gain energy if its below max energy
            if agent_current_fitness[agent_ind]<agent_max_fitness[agent_ind]:
                agent_current_fitness[agent_ind]+=1
        else:
            agent_current_fitness[agent_ind]-=1
            
            #agent will die if fitness is negative
            if agent_current_fitness[agent_ind]<0 and agent_ind in alive_agents:
                agent_loc[agent_ind]=(-1,-1)
                stats['starvation'][timestep]+=1
                alive_agents.remove(agent_ind)
                if agent_ind in dft1_prognosis:
                    dft1_prognosis[agent_ind].append(timestep)
                if agent_ind in dft2_prognosis:
                    dft2_prognosis[agent_ind].append(timestep)
                
                
    return agent_loc, agent_current_fitness, resource_count, resource_num, alive_agents, stats, dft1_prognosis, dft2_prognosis


# In[10]:


#find pairs that can satisfy all conditions in mating season
def handle_mating(agent_loc, agent_gender, agent_reprod, agent_age, age_denom, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
        
    pregnancy=[]

    #get pair in the same grid cell
    duplicates = {}
    for index, value in enumerate(agent_loc):
        if value in duplicates:
            duplicates[value].append(index)
        else:
            duplicates[value] = [index]

    # Filter to only keep the items with more than one occurrence
    matching={k: v for k, v in duplicates.items() if len(v)==2 and k!=(-1,-1)}

    #find pairs satisfying all conditions
    for pair in matching:

        #need both to be different genders
        #has not entered reprod this year
        #older than one year
        if agent_gender[matching[pair][0]]*agent_gender[matching[pair][1]]<0 and agent_reprod[matching[pair][0]]+agent_reprod[matching[pair][1]]==0 and \
        agent_age[matching[pair][0]]//age_denom>=1 and agent_age[matching[pair][1]]//age_denom>=1:

            #add female into pregnancy
            if agent_gender[matching[pair][0]]==-1:
                pregnancy.append(matching[pair][0])
            else:
                pregnancy.append(matching[pair][1])
                
            #update reprod status
            agent_reprod[matching[pair][0]]=1
            agent_reprod[matching[pair][1]]=1
                
    return pregnancy,agent_reprod


# In[11]:


#create new infants
def handle_reproduction(agent_home, pregnancy, agent_current_fitness, agent_loc, agent_age, agent_gender,
                        agent_max_fitness, agent_current_fitness_list, alive_agents,
                        min_fitness, max_fitness, max_offspring, 
                        stats, timestep, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
            
    #reproduction
    stats['mating'][timestep]=len(pregnancy)
    for agent_ind in pregnancy:
                
        #higher fitness lead to higher offsprings
        birth_rate=[0.2]*(max_offspring)
        fitness_percentile = agent_current_fitness[agent_ind] / max_fitness * 100
        ind = round(np.percentile(range(1, max_offspring + 1), fitness_percentile)) - 1
        birth_rate[ind]=0.4
        total_w=sum(birth_rate)
        birth_rate=[br/total_w for br in birth_rate]
                    
        #update new babies attributes
        num_of_babies=np.random.choice(range(1,max_offspring+1),p=birth_rate)  
        stats['newborn'][timestep]+=num_of_babies
                
        # Append new agents
        alive_agents+=[al for al in range(len(agent_loc),len(agent_loc)+num_of_babies)]
        agent_loc+=[agent_loc[agent_ind]]*num_of_babies
        agent_home+=[agent_loc[agent_ind]]*num_of_babies
        agent_age+=[0]*num_of_babies
        agent_gender+=np.random.choice([-1,1],num_of_babies).tolist()
        babies_max_fitness=np.random.randint(low=min_fitness,high=max_fitness,size=num_of_babies).tolist()
        babies_current_fitness=[np.random.randint(low=min_fitness-1,high=bmf) for bmf in babies_max_fitness]
        agent_max_fitness+=babies_max_fitness
        agent_current_fitness+=babies_current_fitness

    return agent_home, agent_loc, agent_age, agent_gender, agent_max_fitness, agent_current_fitness, alive_agents, stats


# In[12]:


#simulate i to d, then e to i, then s to e
def dft1_progression(timestep, dft1_exposed_mat, dft1_infected_mat, alive_agents, dft1_fitness_decline_val,
                        dft1_fitness_decline_rate, agent_current_fitness, agent_max_fitness, agent_age, 
                        agent_loc, stats, dft1_prognosis, dft1_latency_mat, age_denom, max_fitness, 
                        max_dft1_transmission_rate, min_dft1_latency_period, max_dft1_latency_period, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
        
    # Randomly select infected agents to get tumor growth and less fitness
    dft1_exposed = list(set(dft1_exposed_mat[-1]).intersection(alive_agents))
    dft1_infected = list(set(dft1_infected_mat[-1]).intersection(alive_agents))
    aleatoire = np.random.random(size=len(dft1_infected))
    candidates = np.where(np.random.random(size=len(dft1_infected)) < dft1_fitness_decline_rate)[0]
        
    # Modifying agent fitness and potentially causing death
    for i_ind in candidates:
        agent_current_fitness[dft1_infected[i_ind]] -= dft1_fitness_decline_val
        agent_max_fitness[dft1_infected[i_ind]] -= dft1_fitness_decline_val
            
        # Agent will die if fitness is negative
        if agent_current_fitness[dft1_infected[i_ind]] < 0 and dft1_infected[i_ind] in alive_agents:
            agent_loc[dft1_infected[i_ind]] = (-1, -1)
            stats['starvation'][timestep] += 1
            alive_agents.remove(dft1_infected[i_ind])
            dft1_prognosis[dft1_infected[i_ind]].append(timestep)
        
    # Exposed agents progress into infected if it lasts longer than latency period
    for agent_ind in dft1_exposed:
        if timestep - dft1_prognosis[agent_ind][0] >= dft1_latency_mat[agent_ind]:
            dft1_exposed.remove(agent_ind)
            dft1_infected.append(agent_ind)

    # Get multiple agents in the same grid cell
    duplicates = {}
    for index, value in enumerate(agent_loc):
        if value in duplicates:
            duplicates[value].append(index)
        else:
            duplicates[value] = [index]

    # Filter to only keep the items with more than one occurrence
    matching = {k: v for k, v in duplicates.items() if len(v) > 1 and k != (-1, -1)}
    new_dft1_exposed = []
        
    # Disease transmission
    for cluster in matching:
        dft1_infected_cluster = set(matching[cluster]).intersection(dft1_infected)
        if len(dft1_infected_cluster) > 0:
            dft1_susceptible_cluster = list(set(matching[cluster]).intersection(
                np.where(np.array(agent_age) >= age_denom)[0]).difference(dft1_infected).difference(dft1_exposed))
            if len(dft1_susceptible_cluster) > 0:
                benchmarks = [agent_current_fitness[agent_ind] / max_fitness * max_dft1_transmission_rate for agent_ind in dft1_susceptible_cluster]
                aleatoire = np.random.random(size=len(dft1_susceptible_cluster))
                new_dft1_exposed += [dft1_susceptible_cluster[s_ind] for s_ind in np.where(aleatoire <= benchmarks)[0]]
        
    # Create latency period for new exposed class
    aleatoire = np.random.randint(low=min_dft1_latency_period, high=max_dft1_latency_period, size=len(new_dft1_exposed)).astype(np.float64)
        
    # Higher fitness leads to longer latency period
    adj_factors = [agent_current_fitness[agent_ind] / max_fitness for agent_ind in new_dft1_exposed]
    aleatoire *= adj_factors
    aleatoire = np.floor(aleatoire).astype(np.int64)
    aleatoire[aleatoire < min_dft1_latency_period] = min_dft1_latency_period
    for e_ind, agent_ind in enumerate(new_dft1_exposed):
        dft1_prognosis[agent_ind] = [timestep]
        dft1_latency_mat[agent_ind] = aleatoire[e_ind]
        
    # Update exposed and infected mat
    dft1_infected = list(set(dft1_infected).intersection(alive_agents))
    dft1_exposed += new_dft1_exposed
    dft1_exposed_mat[-1] = [agent_ind for agent_ind in dft1_exposed]
    dft1_infected_mat[-1] = [agent_ind for agent_ind in dft1_infected]

    #add lowest latitude for dft1 boundary tracking
    if len(dft1_infected)>0:
        stats['dft1 boundary'][timestep]=max([agent_loc[agent_ind][0] for agent_ind in dft1_infected])
        
    # Return all modified variables
    return agent_current_fitness, agent_max_fitness, alive_agents, dft1_prognosis, dft1_exposed_mat, dft1_infected_mat, stats, dft1_latency_mat


# In[13]:


#simulate i to d, then e to i, then s to e
def dft2_progression(timestep, dft2_exposed_mat, dft2_infected_mat, alive_agents, dft2_fitness_decline_val,
                        dft2_fitness_decline_rate, agent_current_fitness, agent_max_fitness, agent_age, 
                        agent_loc, stats, dft2_prognosis, dft2_latency_mat, age_denom, max_fitness, 
                        max_dft2_transmission_rate, min_dft2_latency_period, max_dft2_latency_period, seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
        
    # Randomly select infected agents to get tumor growth and less fitness
    dft2_exposed = list(set(dft2_exposed_mat[-1]).intersection(alive_agents))
    dft2_infected = list(set(dft2_infected_mat[-1]).intersection(alive_agents))
    aleatoire = np.random.random(size=len(dft2_infected))
    candidates = np.where(np.random.random(size=len(dft2_infected)) < dft2_fitness_decline_rate)[0]
        
    # Modifying agent fitness and potentially causing death
    for i_ind in candidates:
        agent_current_fitness[dft2_infected[i_ind]] -= dft2_fitness_decline_val
        agent_max_fitness[dft2_infected[i_ind]] -= dft2_fitness_decline_val
            
        # Agent will die if fitness is negative
        if agent_current_fitness[dft2_infected[i_ind]] < 0 and dft2_infected[i_ind] in alive_agents:
            agent_loc[dft2_infected[i_ind]] = (-1, -1)
            stats['starvation'][timestep] += 1
            alive_agents.remove(dft2_infected[i_ind])
            dft2_prognosis[dft2_infected[i_ind]].append(timestep)
        
    # Exposed agents progress into infected if it lasts longer than latency period
    for agent_ind in dft2_exposed:
        if timestep - dft2_prognosis[agent_ind][0] >= dft2_latency_mat[agent_ind]:
            dft2_exposed.remove(agent_ind)
            dft2_infected.append(agent_ind)

    # Get multiple agents in the same grid cell
    duplicates = {}
    for index, value in enumerate(agent_loc):
        if value in duplicates:
            duplicates[value].append(index)
        else:
            duplicates[value] = [index]

    # Filter to only keep the items with more than one occurrence
    matching = {k: v for k, v in duplicates.items() if len(v) > 1 and k != (-1, -1)}
    new_dft2_exposed = []
        
    # Disease transmission
    for cluster in matching:
        dft2_infected_cluster = set(matching[cluster]).intersection(dft2_infected)
        if len(dft2_infected_cluster) > 0:
            dft2_susceptible_cluster = list(set(matching[cluster]).intersection(
                np.where(np.array(agent_age) >= age_denom)[0]).difference(dft2_infected).difference(dft2_exposed))
            if len(dft2_susceptible_cluster) > 0:
                benchmarks = [agent_current_fitness[agent_ind] / max_fitness * max_dft2_transmission_rate for agent_ind in dft2_susceptible_cluster]
                aleatoire = np.random.random(size=len(dft2_susceptible_cluster))
                new_dft2_exposed += [dft2_susceptible_cluster[s_ind] for s_ind in np.where(aleatoire <= benchmarks)[0]]
        
    # Create latency period for new exposed class
    aleatoire = np.random.randint(low=min_dft2_latency_period, high=max_dft2_latency_period, size=len(new_dft2_exposed)).astype(np.float64)
        
    # Higher fitness leads to longer latency period
    adj_factors = [agent_current_fitness[agent_ind] / max_fitness for agent_ind in new_dft2_exposed]
    aleatoire *= adj_factors
    aleatoire = np.floor(aleatoire).astype(np.int64)
    aleatoire[aleatoire < min_dft2_latency_period] = min_dft2_latency_period
    for e_ind, agent_ind in enumerate(new_dft2_exposed):
        dft2_prognosis[agent_ind] = [timestep]
        dft2_latency_mat[agent_ind] = aleatoire[e_ind]
        
    # Update exposed and infected mat
    dft2_infected = list(set(dft2_infected).intersection(alive_agents))
    dft2_exposed += new_dft2_exposed
    dft2_exposed_mat[-1] = [agent_ind for agent_ind in dft2_exposed]
    dft2_infected_mat[-1] = [agent_ind for agent_ind in dft2_infected]

    #add lowest latitude for dft2 boundary tracking
    if len(dft2_infected)>0:
        stats['dft2 boundary'][timestep]=min([agent_loc[agent_ind][0] for agent_ind in dft2_infected])
        
    # Return all modified variables
    return agent_current_fitness, agent_max_fitness, alive_agents, dft2_prognosis, dft2_exposed_mat, dft2_infected_mat, stats, dft2_latency_mat


# In[14]:


#simulation of daily model
def run_simulation(num_of_agents, num_of_steps, age_prob, mating_period,
                   all_coordinates, distance_matrix, prob_arr, roadkill_prob, fitness_threshold, monopoly_prob, max_age,
                   age_denom, min_resource, max_resource, min_fitness, max_fitness, max_offspring, max_resource_cap,
                   all_shortest_paths, road, border_row, border_area, beyond, p_inflow, p_outflow,dft1_fitness_decline_val,
                   dft1_intro,max_dft1_transmission_rate,max_dft1_latency_period,min_dft1_latency_period,dft1_fitness_decline_rate,
                   inflow_dft1_infected_rate,dft2_fitness_decline_val,
                   dft2_intro,max_dft2_transmission_rate,max_dft2_latency_period,min_dft2_latency_period,dft2_fitness_decline_rate,ssm,
                   seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
    
    #initialize
    agent_home, agent_loc, agent_max_fitness, agent_current_fitness, agent_mat, resource_mat, agent_age, agent_gender, alive_agents, stats, dft1_prognosis, dft1_exposed_mat, dft1_infected_mat, dft1_latency_mat, dft1_map_mat, dft2_prognosis, dft2_exposed_mat, dft2_infected_mat, dft2_latency_mat, dft2_map_mat=initialize_agents(
        num_of_steps, num_of_agents, all_coordinates, min_fitness, max_fitness, max_age, seed_num)
    
    #iterate thru simulation rounds
    for timestep in range(num_of_steps):
    
        #natural aging
        agent_age, agent_loc, alive_agents, stats, dft1_prognosis, dft2_prognosis=apply_natural_aging(agent_age, agent_current_fitness, alive_agents,
                                                        agent_loc, age_prob, age_denom, max_fitness, timestep, stats, dft1_prognosis, dft2_prognosis, seed_num)
        
        #inflow and outflow
        agent_home, agent_loc, agent_age, agent_gender, agent_max_fitness, agent_current_fitness, alive_agents, stats, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_latency_mat=inflow_outflow(
            agent_home, alive_agents, agent_loc, agent_age, agent_gender, agent_max_fitness, agent_current_fitness,
                   border_row, border_area, beyond, p_inflow, p_outflow, timestep, max_age, max_fitness, min_fitness,
                          stats, dft1_intro, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, inflow_dft1_infected_rate,
                   min_dft1_latency_period, max_dft1_latency_period, age_denom, dft1_latency_mat, seed_num)
        
        # Generate random resource coordinates and capacities
        resource_loc, resource_num, resource_count=generate_resources(all_coordinates,
                                                    prob_arr, min_resource, max_resource, max_resource_cap, resource_mat, seed_num)
        
        #agents move around and consume resources
        agent_loc, agent_current_fitness, resource_count, resource_num, alive_agents, stats, dft1_prognosis, dft2_prognosis=update_agents_status(
            agent_home, alive_agents, agent_loc, agent_current_fitness, agent_max_fitness, road, 
                              roadkill_prob, monopoly_prob, fitness_threshold, distance_matrix, 
                              all_coordinates, all_shortest_paths, resource_count, resource_num, 
                              stats, timestep, dft1_prognosis, dft2_prognosis, seed_num)
        
        #initialize mating situation
        if timestep%age_denom==0:
            agent_reprod=[0]*len(agent_loc)
            
        #add newborn agents into mating track record
        else:
            if len(agent_reprod)<len(agent_loc):
                agent_reprod+=[0]*(len(agent_loc)-len(agent_reprod))
        
        #mating season
        if timestep%age_denom in mating_period:
            
            #find pair
            pregnancy,agent_reprod=handle_mating(agent_loc, agent_gender, agent_reprod, agent_age, age_denom, seed_num)
            
            #reproduction
            agent_home, agent_loc, agent_age, agent_gender, agent_max_fitness, agent_current_fitness, alive_agents, stats=handle_reproduction(
                    agent_home, pregnancy,agent_current_fitness, agent_loc, agent_age, agent_gender,
                            agent_max_fitness, agent_current_fitness, alive_agents,
                            min_fitness, max_fitness, max_offspring, 
                            stats, timestep, seed_num)  
        
        # Disease progression
        if timestep > dft1_intro:
            agent_current_fitness, agent_max_fitness, alive_agents, dft1_prognosis, dft1_exposed_mat, dft1_infected_mat, stats, dft1_latency_mat=dft1_progression(
                timestep, dft1_exposed_mat, dft1_infected_mat, alive_agents, dft1_fitness_decline_val,
                            dft1_fitness_decline_rate, agent_current_fitness, agent_max_fitness, agent_age, 
                            agent_loc, stats, dft1_prognosis, dft1_latency_mat, age_denom, max_fitness, 
                            max_dft1_transmission_rate, min_dft1_latency_period, max_dft1_latency_period, seed_num)
        
            
        # Disease intro
        if timestep == dft2_intro:
            
            #introduce patient zero and randomly assign its infected date based on latency period range
            bottom_parts=max([al[0] for al in agent_loc])
            bottom_locs=[al for al in agent_loc if al[0]==bottom_parts]
            loc_zero=bottom_locs[np.random.randint(low=0,high=len(bottom_locs))]
            patient_zero=agent_loc.index(loc_zero)
            dft2_prognosis[patient_zero]=[]
            dft2_prognosis[patient_zero].append(timestep-np.random.randint(min_dft2_latency_period, max_dft2_latency_period))
            dft2_infected_mat.append([patient_zero])                   
                
        elif timestep > dft2_intro:
            agent_current_fitness, agent_max_fitness, alive_agents, dft2_prognosis, dft2_exposed_mat, dft2_infected_mat, stats, dft2_latency_mat=dft2_progression(
                timestep, dft2_exposed_mat, dft2_infected_mat, alive_agents, dft2_fitness_decline_val,
                            dft2_fitness_decline_rate, agent_current_fitness, agent_max_fitness, agent_age, 
                            agent_loc, stats, dft2_prognosis, dft2_latency_mat, age_denom, max_fitness, 
                            max_dft2_transmission_rate, min_dft2_latency_period, max_dft2_latency_period, seed_num)
            
        else:
            dft2_infected_mat.append([])
                        
        
#             #integrity test
#             dft2_exposed=dft2_exposed_mat[-1]
#             dft2_infected=dft2_infected_mat[-1]
#             zz=[i for i in dft2_prognosis if len(dft2_prognosis[i])<2]
#             if sorted(zz)!=sorted(dft2_exposed+dft2_infected):
#                 print(timestep)
#                 print(len(set(dft2_exposed)),len(dft2_exposed))
#         if len(agent_loc)!=len(agent_home):
#             print(timestep)
            
        #update agent position
        #agent_mat.append([loc_coordinates for loc_coordinates in agent_loc])
        stats['total alive'][timestep]=len(alive_agents)
        
        #update disease stats
        stats['dft1 exposed'][timestep]=len(dft1_exposed_mat[-1])
        stats['dft1 infected'][timestep]=len(dft1_infected_mat[-1])
        stats['dft1 susceptible'][timestep]=len(alive_agents)-len(dft1_exposed_mat[-1])-len(dft1_infected_mat[-1])
        stats['dft2 exposed'][timestep]=len(dft2_exposed_mat[-1])
        stats['dft2 infected'][timestep]=len(dft2_infected_mat[-1])
        stats['dft2 susceptible'][timestep]=len(alive_agents)-len(dft2_exposed_mat[-1])-len(dft2_infected_mat[-1])
        stats['average fitness'][timestep]=sum(agent_current_fitness)/len(agent_current_fitness)
        stats['coinfection'][timestep]=len(set(dft1_infected_mat[-1]).intersection(set(dft2_infected_mat[-1])))
        
        if timestep//age_denom in ssm:
            
            yearstep=timestep//age_denom
            
            #compute dft1 infected proportion
            dft1_map_denominator=np.zeros(len(ssm[yearstep]))
            dft1_map_numerator=np.zeros(len(ssm[yearstep]))
            for agent_ind in alive_agents:
                if agent_loc[agent_ind] in ssm[yearstep]:
                    dft1_map_denominator[ssm[yearstep].index(agent_loc[agent_ind])]+=1
            for agent_ind in dft1_infected_mat[-1]:
                if agent_loc[agent_ind] in ssm[yearstep]:
                    dft1_map_numerator[ssm[yearstep].index(agent_loc[agent_ind])]+=1

            #avoid zero division
            dft1_map_denominator[dft1_map_denominator==0]=0.00001
            dft1_map_mat.append((dft1_map_numerator/dft1_map_denominator).tolist())

            #compute dft2 infected proportion
            dft2_map_denominator=np.zeros(len(ssm[yearstep]))
            dft2_map_numerator=np.zeros(len(ssm[yearstep]))
            for agent_ind in alive_agents:
                if agent_loc[agent_ind] in ssm[yearstep]:
                    dft2_map_denominator[ssm[yearstep].index(agent_loc[agent_ind])]+=1
            for agent_ind in dft2_infected_mat[-1]:
                if agent_loc[agent_ind] in ssm[yearstep]:
                    dft2_map_numerator[ssm[yearstep].index(agent_loc[agent_ind])]+=1

            #avoid zero division
            dft2_map_denominator[dft2_map_denominator==0]=0.00001
            dft2_map_mat.append((dft2_map_numerator/dft2_map_denominator).tolist())

    return stats, agent_mat, resource_mat, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_map_mat, dft2_exposed_mat, dft2_infected_mat, dft2_prognosis, dft2_map_mat


# In[15]:


#divide input variable range equally then stratify
#if variable range is too narrow to satisfy number of simulations
#randomly draw interger from that range
def lhs_preparation(min_num,max_num,num_of_simu,seed_num=None):
    
    #random number control
    if seed_num is None:
        pass
    else:
        np.random.seed(seed_num)
        
    #stratify
    unstratified=np.arange(min_num,max_num,(max_num-min_num)/num_of_simu)
    np.random.shuffle(unstratified)
    
    if len(unstratified)>num_of_simu:
        unstratified=unstratified[:num_of_simu]
    elif len(unstratified)<num_of_simu:
        unstratified=np.random.choice(unstratified,size=num_of_simu)
    else:
        pass
    return unstratified


# ### definition

# In[16]:


#Create a list of all possible coordinates
df=pd.read_csv('road.csv',header=None)

indices = np.where(np.array(df) == 0)
all_coordinates = list(zip(indices[0],indices[1],))

#road that creates higher probability of resource on roadside
#road also creates roadkill probability
indices = np.where(np.array(df) == 128)
road = list(zip(indices[0],indices[1],))


# In[17]:


#random number control
seed_num=None
np.random.seed(seed_num)


# #### pre-determined variables

# In[18]:


#max density obtained from poems
max_density=1.5
max_capacity=max_density*len(all_coordinates)

#max capacity of one resource
max_resource_cap=5

#number of initial agents
num_of_agents=500

#max offspring per agent
max_offspring=4

#define the border area in map which enable agents to travel beyond the map
border_row=5
border_area=[i for i in all_coordinates if i[0]<border_row]

#define the number of agents beyond the map
beyond=int(len(df)*border_row/4)

#mirror inflow and outflow prob
p_inflow=0.01
p_outflow=0.01


# #### disease variables

# In[19]:


#timestep where first dft1 case went in
dft1_intro=365

#transmission rate fluctuates based upon each agent's fitness
#higher fitness leads to higher transmission rate
max_dft1_transmission_rate=0.75

#latency period fluctuates based upon each agent's fitness
#higher fitness leads to longer latency period
max_dft1_latency_period=300
min_dft1_latency_period=100

#random prob to determine how rapidly the fitness would decline after being infected
dft1_fitness_decline_rate=0.3

#how much fitness is declined for infected agent
dft1_fitness_decline_val=1

#what proportion of immigration is exposed and infected after patient zero
inflow_dft1_infected_rate=0.66


# In[20]:


#timestep where first dft2 case went in
dft2_intro=1095

#transmission rate fluctuates based upon each agent's fitness
#higher fitness leads to higher transmission rate
max_dft2_transmission_rate=0.75

#latency period fluctuates based upon each agent's fitness
#higher fitness leads to longer latency period
max_dft2_latency_period=300
min_dft2_latency_period=100

#random prob to determine how rapidly the fitness would decline after being infected
dft2_fitness_decline_rate=0.3

#how much fitness is declined for infected agent
dft2_fitness_decline_val=1


# #### population variables

# In[21]:


#min number of resources
min_resource=400

#max number of resources
max_resource=600

#roadkill probability for the agent
roadkill_prob=0.0005

#max and min fitness
#fitness defines max distance an agent can travel in one time step
#also defines the time steps which agent can survive without resource
max_fitness=20
min_fitness=10

#if current fitness larger than fitness threshold
#there is possibility for monopoly of the resource
fitness_threshold=int(0.9*max_fitness)
monopoly_prob=0.1

#the probability of natural mortality at each age
age_prob=[0.0003,0.0005,0.0005,0.001,0.002,0.003, 1]

#maximum age
max_age=2190

#define how many timesteps equal to one year in age
age_denom=max_age/(len(age_prob)-1)

#mating period is in the format of timestep%age_denom
mating_period=[i for i in range(90,150)]

#get simulation summary mwtrics
valid=pd.read_csv('disease.csv')

#baseyear
baseyear=2008

#etl
valid['timestep']=valid['year']-baseyear
valid['id']=valid['id'].apply(lambda x:ast.literal_eval(x))
                  
ssm={}
for i in valid.index:
    if valid.at[i,'timestep'] in ssm:
        ssm[valid.at[i,'timestep']].append(valid.at[i,'id'])
    else:
        ssm[valid.at[i,'timestep']]=[valid.at[i,'id']]

# #### simulation variables

# In[31]:


#number of timesteps in one simulation
num_of_steps=5500

#number of simulations
num_of_simu=5000
#number of cpu cores working
num_parallel=20

lhs={}

lhs['min_resource']=lhs_preparation(100,300,num_of_simu,seed_num).astype(int)
lhs['max_resource']=lhs_preparation(400,600,num_of_simu,seed_num).astype(int)
lhs['max_fitness']=lhs_preparation(11,max_fitness,num_of_simu,seed_num).astype(int)
lhs['min_fitness']=lhs_preparation(2,min_fitness,num_of_simu,seed_num).astype(int)
lhs['max_dft1_transmission_rate']=lhs_preparation(0.1,0.9,num_of_simu,seed_num)
lhs['max_dft1_latency_period']=lhs_preparation(200,540,num_of_simu,seed_num).astype(int)
lhs['min_dft1_latency_period']=lhs_preparation(min_dft1_latency_period,180,num_of_simu,seed_num).astype(int)
lhs['dft1_fitness_decline_rate']=lhs_preparation(0.1,0.9,num_of_simu,seed_num)
lhs['dft1_fitness_decline_val']=lhs_preparation(1,5,num_of_simu,seed_num).astype(int)
lhs['max_dft2_transmission_rate']=lhs_preparation(0.1,0.9,num_of_simu,seed_num)
lhs['max_dft2_latency_period']=lhs_preparation(200,540,num_of_simu,seed_num).astype(int)
lhs['min_dft2_latency_period']=lhs_preparation(min_dft2_latency_period,180,num_of_simu,seed_num).astype(int)
lhs['dft2_fitness_decline_rate']=lhs_preparation(0.1,0.9,num_of_simu,seed_num)
lhs['dft2_fitness_decline_val']=lhs_preparation(1,5,num_of_simu,seed_num).astype(int)
lhs['dft1_intro']=lhs_preparation(730,1460,num_of_simu,seed_num).astype(int)
lhs['dft2_intro']=lhs_preparation(365,1825,num_of_simu,seed_num).astype(int)



# ### preparation

# In[23]:


#get possible neighbors
roadside=set([neighbor for point in road for neighbor in get_neighbors(point[0],point[1])])

#remove road itself
roadside=roadside.difference(set(road))

#remove out of bound
roadside=[i for i in roadside if i[0]<len(df) and i[1]<len(df) and i[0]>=0 and i[1]>=0]


# In[24]:


# # Create an empty distance matrix
# num_coords = len(all_coordinates)
# distance_matrix = np.zeros((num_coords, num_coords))

# # Populate the distance matrix with Euclidean distances
# for i in range(num_coords):
#     for j in range(i + 1, num_coords):  # To avoid redundant calculations
#         dist = euclidean_distance(all_coordinates[i], all_coordinates[j])
#         distance_matrix[i, j] = dist
#         distance_matrix[j, i] = dist  # The distance matrix is symmetric

# Now `distance_matrix[i, j]` gives the Euclidean distance between all_coordinates[i] and all_coordinates[j]

# # Precompute shortest paths from all agents to all resources
# all_shortest_paths = [[[] for _ in range(num_coords)] for _ in range(num_coords)]

# # Populate the shortest path
# for i in range(num_coords):
#     for j in range(i + 1, num_coords):  # To avoid redundant calculations
#         all_shortest_paths[j][i] = shortest_path(all_coordinates[i][0],all_coordinates[i][1],
#                                                  all_coordinates[j][0],all_coordinates[j][1],)
#         all_shortest_paths[i][j] = all_shortest_paths[j][i]

# # Save it to a file
# variables={}
# variables['distance_matrix']=distance_matrix
# variables['all_shortest_paths']=all_shortest_paths
# with open('variables.pkl', 'wb') as f:
#     pickle.dump(variables, f)


# In[25]:


#load precalculated memoization variables
with open('variables.pkl', 'rb') as f:
    variables = pickle.load(f)

distance_matrix=variables['distance_matrix']
all_shortest_paths=variables['all_shortest_paths']


# In[26]:


#higher resource probability for roadside
prob=[]
for i in range(len(all_coordinates)):
    if all_coordinates[i] in roadside:
        prob.append(8)
    else:
        prob.append(4)        
prob_arr=[i/sum(prob) for i in prob]


# ### simulation

# In[27]:


# stats, agent_mat, resource_mat, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_map_mat, dft2_exposed_mat, dft2_infected_mat, dft2_prognosis, dft2_map_mat=run_simulation(
#                     num_of_agents, num_of_steps, age_prob, mating_period,
#                    all_coordinates, distance_matrix, prob_arr, roadkill_prob, fitness_threshold, monopoly_prob, max_age,
#                    age_denom, min_resource, max_resource, min_fitness, max_fitness, max_offspring, max_resource_cap,
#                    all_shortest_paths, road, border_row, border_area, beyond, p_inflow, p_outflow,
#                     dft1_fitness_decline_val,dft1_intro,max_dft1_transmission_rate,max_dft1_latency_period,
#                     min_dft1_latency_period,dft1_fitness_decline_rate,inflow_dft1_infected_rate,
#                   dft2_fitness_decline_val,dft2_intro,max_dft2_transmission_rate,
#                     max_dft2_latency_period,min_dft2_latency_period,dft2_fitness_decline_rate)


# In[28]:




# In[29]:


# #integrity test
# for i in dft2_prognosis:
#     if len(dft2_prognosis[i])==2:
#         for j in agent_mat:
#             if len(j)>i and j[i]==(-1,-1):
#                 print(agent_mat.index(j)==dft2_prognosis[i][1]+1)
#                 break


# In[33]:


# stats_dict={};agent_dict={};resource_dict={}; dft1_exposed_dict={}; dft1_infected_dict={}; dft1_prognosis_dict={}; dft1_map_dict={}
# dft2_exposed_dict={}; dft2_infected_dict={}; dft2_prognosis_dict={}; dft2_map_dict={}
# for i in range(num_of_simu):
#     print(i)
#     stats, agent_mat, resource_mat, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_map_mat, dft2_exposed_mat, dft2_infected_mat, dft2_prognosis, dft2_map_mat=run_simulation(
#         num_of_agents, num_of_steps, age_prob, mating_period,
#                    all_coordinates, distance_matrix, prob_arr, roadkill_prob, fitness_threshold, monopoly_prob, max_age,
#                    age_denom, lhs['min_resource'][i], lhs['max_resource'][i], lhs['min_fitness'][i], lhs['max_fitness'][i], 
#                 max_offspring, max_resource_cap,
#                    all_shortest_paths, road, border_row, border_area, beyond, p_inflow, p_outflow,
#                     lhs['dft1_fitness_decline_val'][i],dft1_intro,lhs['max_dft1_transmission_rate'][i],
#                 lhs['max_dft1_latency_period'][i],lhs['min_dft1_latency_period'][i],
#                         lhs['dft1_fitness_decline_rate'][i],inflow_dft1_infected_rate,
#                   lhs['dft2_fitness_decline_val'][i],dft2_intro,lhs['max_dft2_transmission_rate'][i],
#                     lhs['max_dft2_latency_period'][i],lhs['min_dft2_latency_period'][i],lhs['dft2_fitness_decline_rate'][i]
#         )
    
#     stats_dict[i]=stats
# #     agent_dict[i]=agent_mat
# #     resource_dict[i]=resource_mat
# #     dft1_exposed_dict[i]=dft1_exposed_mat
# #     dft1_infected_dict[i]=dft1_infected_mat
#     dft1_prognosis_dict[i]=dft1_prognosis
# #     dft1_map_dict[i]=dft1_map_mat
# #     dft2_exposed_dict[i]=dft2_exposed_mat
# #     dft2_infected_dict[i]=dft2_infected_mat
#     dft2_prognosis_dict[i]=dft2_prognosis
# #     dft2_map_dict[i]=dft2_map_mat


# In[ ]:


import concurrent.futures
import datetime as dt
print(f'running simulations of {num_of_simu}')
print(dt.datetime.now())
# Function to run the simulation for each iteration
def run_simulation_task(i, num_of_agents, num_of_steps, age_prob, mating_period,
                   all_coordinates, distance_matrix, prob_arr, roadkill_prob, fitness_threshold, monopoly_prob, max_age,
                   age_denom, min_resource, max_resource, min_fitness, max_fitness, max_offspring, max_resource_cap,
                   all_shortest_paths, road, border_row, border_area, beyond, p_inflow, p_outflow,
                    dft1_fitness_decline_val,dft1_intro,max_dft1_transmission_rate,max_dft1_latency_period,
                    min_dft1_latency_period,dft1_fitness_decline_rate,inflow_dft1_infected_rate,
                  dft2_fitness_decline_val,dft2_intro,max_dft2_transmission_rate,
                    max_dft2_latency_period,min_dft2_latency_period,dft2_fitness_decline_rate, ssm):
    
    stats, agent_mat, resource_mat, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_map_mat, dft2_exposed_mat, dft2_infected_mat, dft2_prognosis, dft2_map_mat = run_simulation(
        num_of_agents, num_of_steps, age_prob, mating_period,
                   all_coordinates, distance_matrix, prob_arr, roadkill_prob, fitness_threshold, monopoly_prob, max_age,
                   age_denom, min_resource, max_resource, min_fitness, max_fitness, max_offspring, max_resource_cap,
                   all_shortest_paths, road, border_row, border_area, beyond, p_inflow, p_outflow,
                    dft1_fitness_decline_val,dft1_intro,max_dft1_transmission_rate,max_dft1_latency_period,
                    min_dft1_latency_period,dft1_fitness_decline_rate,inflow_dft1_infected_rate,
                  dft2_fitness_decline_val,dft2_intro,max_dft2_transmission_rate,
                    max_dft2_latency_period,min_dft2_latency_period,dft2_fitness_decline_rate, ssm
    )

    return i, stats, agent_mat, resource_mat, dft1_exposed_mat, dft1_infected_mat, dft1_prognosis, dft1_map_mat, dft2_exposed_mat, dft2_infected_mat, dft2_prognosis, dft2_map_mat

# Initialize dictionaries
stats_dict = {}
dft1_exposed_dict = {}
dft1_infected_dict = {}
dft1_prognosis_dict = {}
dft1_map_dict = {}
dft2_exposed_dict = {}
dft2_infected_dict = {}
dft2_prognosis_dict = {}
dft2_map_dict = {}

# Run the simulation in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel) as executor:
    # Submit all tasks for parallel execution
    futures = [
        executor.submit(run_simulation_task, i, num_of_agents, num_of_steps, age_prob, mating_period,
                   all_coordinates, distance_matrix, prob_arr, roadkill_prob, fitness_threshold, monopoly_prob, max_age,
                   age_denom, lhs['min_resource'][i], lhs['max_resource'][i], lhs['min_fitness'][i], lhs['max_fitness'][i], 
                max_offspring, max_resource_cap,
                   all_shortest_paths, road, border_row, border_area, beyond, p_inflow, p_outflow,
                    lhs['dft1_fitness_decline_val'][i],lhs['dft1_intro'][i],lhs['max_dft1_transmission_rate'][i],
                lhs['max_dft1_latency_period'][i],lhs['min_dft1_latency_period'][i],
                        lhs['dft1_fitness_decline_rate'][i],inflow_dft1_infected_rate,
                  lhs['dft2_fitness_decline_val'][i],lhs['dft2_intro'][i],lhs['max_dft2_transmission_rate'][i],
                    lhs['max_dft2_latency_period'][i],lhs['min_dft2_latency_period'][i],lhs['dft2_fitness_decline_rate'][i], ssm)
        for i in range(num_of_simu)
    ]
    
    # Process the results as they complete
    for future in concurrent.futures.as_completed(futures):
        i, stats, _, _, _, _, dft1_prognosis, dft1_map_mat, _, _,dft2_prognosis,dft2_map_mat = future.result()
        stats_dict[i] = stats
        dft1_prognosis_dict[i] = dft1_prognosis
        dft2_prognosis_dict[i] = dft2_prognosis
        dft1_map_dict[i] = dft1_map_mat
        dft2_map_dict[i] = dft2_map_mat

print(dt.datetime.now())


# In[ ]:


# with open('agents.pkl', 'wb') as f:
#     pickle.dump(agent_dict,f)

# with open('resources.pkl', 'wb') as f:
#     pickle.dump(resource_dict,f)

with open('stats.pkl', 'wb') as f:
    pickle.dump(stats_dict,f)

with open('lhs.pkl', 'wb') as f:
    pickle.dump(lhs,f)
    
# with open('dft1_e.pkl', 'wb') as f:
#     pickle.dump(dft1_exposed_dict,f)
    
# with open('dft1_i.pkl', 'wb') as f:
#     pickle.dump(dft1_infected_dict,f)
    
with open('dft1_p.pkl', 'wb') as f:
    pickle.dump(dft1_prognosis_dict,f)
    
with open('dft1_m.pkl', 'wb') as f:
    pickle.dump(dft1_map_dict,f)

# with open('dft2_e.pkl', 'wb') as f:
#     pickle.dump(dft2_exposed_dict,f)
    
# with open('dft2_i.pkl', 'wb') as f:
#     pickle.dump(dft2_infected_dict,f)
    
with open('dft2_p.pkl', 'wb') as f:
    pickle.dump(dft2_prognosis_dict,f)
    
with open('dft2_m.pkl', 'wb') as f:
    pickle.dump(dft2_map_dict,f)

