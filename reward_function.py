from pettingzoo.mpe import  simple_tag_v2




ep_len=25
new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)
