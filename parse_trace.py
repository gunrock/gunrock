#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import re

def milliseconds(nanoseconds):
    return nanoseconds / 1000000.0



class TraceItem:
    def __init__(self, name, queue_name, queued, submit, start, end, index, dependencies):
        self.name = name
        self.queue_name = queue_name
        self.queued = queued
        self.submit = submit
        self.start = start
        self.end = end
        self.index = index
        self.dependencies = dependencies

    def duration_ms(self):
        return milliseconds(self.end - self.start)

    def start_ms(self):
        return milliseconds(self.start)
    
    def end_ms(self):
        return milliseconds(self.end)

    def handle(self):
        return (self.queue_name,self.name)

    

def parse(file):
    pattern = re.compile(r'([^@:]*)(@[^@:]*):(\d+):(\d+):(\d+):(\d+):(\d+):([0-9|]*)')
    
    items = []

    for line in file.readlines():
        match = pattern.match(line)

        if not match:
            match = pattern.match(line.strip()+":0:")

            if not match:
                print('Input file format mismatch.')
                exit(1)
        
        item = TraceItem(match.group(1),
                         match.group(2)[1:],
                         int(match.group(3)), 
                         int(match.group(4)), 
                         int(match.group(5)), 
                         int(match.group(6)),
                         int(match.group(7)),
                         [int(s) for s in match.group(8).split('|') if s])
            
        items.append(item)

    items.reverse()
        
    return items



def find_time_range(trace_items):
    tmin = min((ti.start for ti in trace_items))
    tmax = max((ti.end for ti in trace_items))
    
    return tmin,tmax



def offset_time_ranges(trace_items, tmin):
    for ti in trace_items:
        ti.queued -= tmin
        ti.submit -= tmin
        ti.start  -= tmin
        ti.end    -= tmin
        


def create_queue_list(trace_items):
    queue_list = []

    queue_dict = {}
    inserted = {('','')}
    
    for qn,n in [(i.queue_name, i.name) for i in trace_items]:
        ns = None

        if qn in queue_dict:
            ns = queue_dict[qn]
        else:
            ns = []
            queue_dict[qn] = ns
            queue_list.append((qn,ns))
            
        if (qn,n) not in inserted:
            ns.append(n)
            inserted.add((qn,n))

    return queue_list

