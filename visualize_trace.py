#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import colorsys

import random
import math

import cairo

from collections import defaultdict
import math

from optparse import OptionParser

from parse_trace import *

def generate_colorscheme(queue_list):
    color_scheme = {}
    queue_cnt = len(queue_list)
    command_cnt = max([len(ns) for qn,ns in queue_list])
    offset = random.random()
    
    for qi,(qn,ns) in enumerate(queue_list):
        for ni,n in enumerate(ns):
            h = 1 / (1.5*queue_cnt) * (1.5*qi + ni/(command_cnt-1)) + offset
            v = 0.65 + 0.35 * ni/(command_cnt-1)
            s = 0.65 + 0.35 * ni/(command_cnt-1)
            color_scheme[(qn,n)] = colorsys.hsv_to_rgb(h,s,v)
            
    return color_scheme



def show_color_scheme(queue_list, color_scheme):
    with open('colorscheme.html', 'w') as outfile:
        outfile.write('<head>\n'
                      '  <title>Hello</title>\n'
                      '</head>\n'
                      '<body>\n')

        for qn,ns in queue_list.items():
            for n in ns:
                c = [int(255.9999*c) for c in color_scheme[qn,n]]
                
                outfile.write('  <span style="background-color:#%02x%02x%02x">%s</span>\n' % (c[0],c[1],c[2],n))
            outfile.write('  <br>\n');
        
        outfile.write('</body>')


def mm(mm):
    return mm / 25.4 * 72.0

def setup_context(surface, padding):
    context = cairo.Context(surface)
    context.scale(1.0 / 25.4 * 72.0, 1.0 / 25.4 * 72.0)
    context.set_line_width(0.25)
    context.translate(padding, padding)
    context.save()

    return context

def offset(w,h, f, context, *params):
    context.save()
    context.translate(w,h)

    f(context, *params)

    context.restore()
    
def aligned_text(ctx, txt, fx, fy, scale=1.0):
    xb, yb, w, h, xa, ya = ctx.text_extents(txt)

    ctx.save()
    ctx.scale(scale, scale)
    ctx.rel_move_to(-w*fx, h*fy)
    ctx.show_text(txt)
    ctx.restore()

def rounded_rect(ctx, x,y,w,h, r):
    if h < r * 2:
        r = h/2
    if w < r * 2:
        r = w/2
    
    ctx.move_to(x, y+r)
    ctx.arc(x+r,y+r, r, math.pi, math.pi * 1.5)
    ctx.line_to(x+w-r, y)
    ctx.arc(x+w-r,y+r, r, math.pi * -0.5, math.pi * 0.0)
    ctx.line_to(x+w, y+h-r)
    ctx.arc(x+w-r, y+h-r, r, math.pi * 0.0, math.pi * 0.5)
    ctx.line_to(x+r, y+h)
    ctx.arc(x+r, y+h-r, r, math.pi * 0.5, math.pi * 1.0)
    #ctx.line_to(x, y+r)
    ctx.close_path()
    
def draw_legend(ctx, queue_list, font, height_per_entry):
    
    queue_indent = 5

    legend_height = 0

    ctx.set_scaled_font(font)
    
    for qn,ns in queue_list:
        ctx.move_to(0, legend_height + height_per_entry/2)
        aligned_text(ctx, qn, 0.0, 0.5)
        
        legend_height += height_per_entry
        for n in ns:
            ctx.move_to(5, legend_height + height_per_entry/2)
            aligned_text(ctx, n, 0.0, 0.5)
        
            legend_height += height_per_entry


            
def draw_grid_back(ctx, graph_width, graph_height, start, stop):

    increment = graph_width/((stop-start)*10)
    
    for i in range(0, math.ceil(graph_width/increment)):

        if i%10 == 0:
            ctx.set_source_rgb(0.75,0.75,0.75)
        elif i%5 == 0:
            ctx.set_source_rgb(0.87,0.87,0.87)
        else:
            ctx.set_source_rgb(0.95,0.95,0.95)
        
        ctx.move_to(i*increment,0)
        ctx.rel_line_to(0,graph_height)
        ctx.stroke()

        ctx.stroke()


def digit_cnt(f):
    f = f-math.floor(f)

    d = 0
    while f > 0.00001:
        f *= 10
        f = f-math.floor(f)
        d += 1

    return d
        
def draw_grid_front(ctx, graph_width, graph_height, start, stop, scalefactor):

    increment = graph_width/((stop-start)*10)
    ctx.set_source_rgb(0.3,0.3,0.3)

    digits = digit_cnt(scalefactor)
    
    for i in range(0, math.ceil((graph_width+0.0001)/increment)):
        
        nub_len = 1.5

        if i%10 == 0:
            nub_len = 3
            time = float((i/10 + start) * scalefactor)
            ctx.move_to(i*increment, graph_height + 4)
            aligned_text(ctx, ("%" + (".%df ms" % digits)) % time, 0.5, 1.0, 0.35);
        
        ctx.move_to(i*increment,0)
        ctx.rel_line_to(0, -nub_len);
        
        ctx.move_to(i*increment,graph_height)
        ctx.rel_line_to(0, nub_len);

        ctx.stroke()

    
    ctx.set_source_rgb(0.3,0.3,0.3)
    ctx.rectangle(0,0,graph_width, graph_height)
    ctx.stroke()


    
def calc_command_placement(queue_list, command_height, height_per_entry, font):
    legend_width = 0
    queue_indent = 5

    legend_height = 0
    
    for qn,ns in queue_list:
        _,_,w,_,_,_ = font.text_extents(qn)
        legend_width = max(legend_width, w)
        command_height[qn] = legend_height
        legend_height += height_per_entry
        for n in ns:
            _,_,w,_,_,_ = font.text_extents(n)
            legend_width = max(legend_width, w + queue_indent)
            command_height[qn,n] = legend_height
            legend_height += height_per_entry

    return legend_width, legend_height



def draw_items(ctx, trace_items, height_per_entry, width_per_ms, color_scheme, command_height_dict):

    ctx.set_line_width(0.25)
    for ti in trace_items:
        ctx.set_source_rgb(*color_scheme[ti.handle()])
        rounded_rect(ctx,
                     ti.start_ms()*width_per_ms, command_height_dict[ti.handle()]+0.5,
                     ti.duration_ms()*width_per_ms, height_per_entry-1, .5)
        ctx.fill()
        ctx.set_source_rgb(*(c*0.5 for c in color_scheme[ti.handle()]))
        rounded_rect(ctx,
                     ti.start_ms()*width_per_ms, command_height_dict[ti.handle()]+0.5,
                     ti.duration_ms()*width_per_ms, height_per_entry-1, .5)        
        ctx.stroke()

def draw_dependencies(ctx, trace_items, height_per_entry, width_per_ms, color_scheme, command_height_dict):

    trace_index = {ti.index:ti for ti in trace_items}

    ctx.set_source_rgba(0.2,0.2,1,1)
    for e in trace_items:
        for d in e.dependencies:
            s = trace_index[d]
            ctx.move_to(s.end_ms()*width_per_ms, command_height_dict[s.handle()]+height_per_entry*0.5)
            ctx.line_to(e.start_ms()*width_per_ms, command_height_dict[e.handle()]+height_per_entry*0.5)
            ctx.stroke()
            
        

def draw_trace(trace_items, outfilename, show_dependencies, scalefactor):
    tmin,tmax = find_time_range(traceitems)
    offset_time_ranges(traceitems, tmin)   
    tmin,tmax = find_time_range(traceitems)
    
    tmin_ms = milliseconds(tmin)
    tmax_ms = milliseconds(tmax)
    dur_ms = tmax_ms - tmin_ms
        
    queue_list = create_queue_list(traceitems)
    color_scheme = generate_colorscheme(queue_list)

    padding = 10
    height_per_entry = 7
    width_per_ms = 50

    ms_per_page = 8

    command_height_dict = {}
    font = cairo.ScaledFont(cairo.ToyFontFace('FreeSans'), cairo.Matrix(4,0, 0,4, 0,0), cairo.Matrix(), cairo.FontOptions())
    legend_width, legend_height = calc_command_placement(queue_list, command_height_dict, height_per_entry, font)
    
    #graph_width = milliseconds(tmax-tmin) * width_per_ms
    graph_width = ms_per_page * width_per_ms
    graph_height = legend_height
    
    width  = graph_width  + 3*padding + legend_width
    height = graph_height + 2*padding
    
    surface = cairo.PDFSurface(outfilename, mm(width), mm(height))

    context = setup_context(surface, padding)

    for i in range(0,math.ceil(dur_ms/ms_per_page)):
        tmin_p = tmin_ms + i * ms_per_page
        tmax_p = tmin_p + min(max(2, tmax_ms-tmin_p), ms_per_page)

        graph_width = (tmax_p-tmin_p) * width_per_ms

        offset(0,0, draw_legend, context, queue_list, font, height_per_entry)

        offset(legend_width+padding,0, draw_grid_back, context, graph_width, graph_height, tmin_p, tmax_p)
        context.save()
        context.rectangle(legend_width+padding,0,graph_width, graph_height)
        context.clip()
        offset(legend_width+padding-tmin_p*width_per_ms,0, draw_items, context, trace_items,
               height_per_entry, width_per_ms, color_scheme, command_height_dict)
        if show_dependencies:
            offset(legend_width+padding-tmin_p*width_per_ms,0, draw_dependencies, context, trace_items,
                height_per_entry, width_per_ms, color_scheme, command_height_dict)
        context.restore()
        offset(legend_width+padding,0, draw_grid_front, context, graph_width, graph_height, tmin_p, tmax_p, scalefactor)

        context.show_page()
        
    surface.finish()


def parse_args():
    parser = OptionParser(usage='Usage: %prog [options] <tracefile> <outfile>')
    parser.add_option('-d', '--dependencies',
                      action='store_true', dest='show_dependencies', default=False,
                      help='Show command dependencies in output')
    parser.add_option('-s', '--scalefactor',
                      type='float', dest='scalefactor', default=1.0,
                      help='Time scale factor')
    options, args = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
        exit(0)
    elif len(args) > 2:
        parser.print_help()
        exit(1)

    return options, args[0], args[1]

        
if __name__=='__main__':

    options, infile, outfile = parse_args()

    with open(infile, 'r') as tracefile:
        traceitems = parse(tracefile)

    for ti in traceitems:
        ti.queued /= options.scalefactor
        ti.submit /= options.scalefactor
        ti.start /= options.scalefactor
        ti.end /= options.scalefactor

    draw_trace(traceitems, outfile, options.show_dependencies, options.scalefactor)
    
