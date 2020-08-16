import time
from collections import defaultdict


class Personalized():
    def __init__(self, nx_G, mask, args):
        self.G = nx_G
        self.mask = mask
        self.args = args

    # iterative version
    def dfs(self, start_node, walks_num):
        stack=[]
        stack.append(start_node)
        seen=set()
        seen.add(start_node)
        walks = []
        mask_list = set(self.mask[start_node])
        while (len(stack)>0):
            vertex=stack.pop()
            nodes=self.G[vertex]
            # print("nodes", nodes)
            for w in nodes:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
            if start_node < self.args.user_num:
                # print("user...")
                if vertex > self.args.user_num:
                    if vertex in mask_list:
                        pass
                    else:
                        walks.append(vertex)
                else:
                    pass
            else:
                # print("item...")
                if vertex > self.args.user_num:
                    if vertex in mask_list:
                        pass
                    else:
                        if vertex == start_node:
                            pass
                        else:
                            walks.append(vertex)
                else:
                    pass
            if len(walks) >= walks_num:
                break
        return walks

    # recursiveche version
    # def dfs(self, start_node, walks=[]):
    #     walks.append(start_node)
    #     mask_list = set(self.mask[start_node])
    #     for w in self.G[start_node]:
    #         if w not in walks:
    #             if len(walks) >= walks_num:
    #                 break
    #             if w > user_num:
    #                 if w in mask_list:
    #                     pass
    #                 else:
    #                     if w == start_node:
    #                         pass
    #                     else:
    #                         dfs(G, w, walks)
    #             else:
    #                 pass
    #     return walks
    
    def intermediate(self):
        candidate = defaultdict(list)
        for node in self.G.nodes():
            if node < self.args.user_num:
                pass
            else:
                walk = self.dfs(node, self.args.walks_num)
                candidate[node].extend(walk)
        return candidate


def candidate_choose(nx_Graph, mask, args):
    G = Personalized(nx_Graph, mask, args)
    candidates = G.intermediate()
    return candidates
