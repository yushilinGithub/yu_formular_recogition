# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
#from asyncio.log import logger
import logging
import os
import re
import sys
import io
import glob
import tempfile
import subprocess
import uuid
import shutil
from PIL import Image


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\setmathfont{%s}
\begin{document}
\thispagestyle{empty}
%s
\end{document}
'''

    def __init__(self, math,line, dpi=250, font='Latin Modern Math'):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        self.math = math
        self.dpi = dpi
        self.font = font
        self.line = line
    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False
        try:
            workdir = tempfile.gettempdir()
            #workdir = "picc"
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
       
            with os.fdopen(fd, 'w+') as f: 

                document = self.BASE % (self.font, self.math)

                try:
                    f.write(document)
                except:
                    logging.error("latex file error %s"%(self.math))

            png = self.convert_file(texfile, workdir, return_bytes=return_bytes)

            return png

        finally:
            if os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False):

        try:
                # Generate the PDF file
            cmd = 'xelatex -halt-on-error -output-directory %s %s' % (workdir, infile)

            p = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            sout, serr = p.communicate()
            # Something bad happened, abort
            if p.returncode != 0:
                logging.error('latex error %s %s %s'%(self.line,serr,sout))
                return False

            # Convert the PDF file to PNG's
            pdffile = infile.replace('.tex', '.pdf')
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))


            cmd = 'convert -density %i -colorspace gray %s -quality 90 %s' % (
                self.dpi,
                pdffile,
                pngfile,
            )  # -bg Transparent -z 9
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            p = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sout, serr = p.communicate()
            if p.returncode != 0:
                #raise Exception('PDFpng error', serr, cmd, os.path.exists(pdffile), os.path.exists(infile))
                logging.error('PDFpng error %s,%s,%s,%s,%s'%(self.line,serr,cmd,os.path.exists(pdffile),os.path.exists(infile)))
                return False
            if return_bytes:
                if isinstance(self.math,list) and len(self.math) > 1:
                    png = [open(pngfile.replace('.png', '')+'-%i.png' % i, 'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace('.png', '')+'.png', 'rb').read()]
                return png
            else:
                if isinstance(self.math,list) and len(self.math) > 1:
                    return [(pngfile.replace('.png', '')+'-%i.png' % i) for i in range(len(self.math))]
                else:
                    return pngfile.replace('.png', '')+'.png'
        finally:
            # Cleanup temporaries
            basefile = infile.replace('.tex', '')
            tempext = ['.aux', '.pdf', '.log']
            if return_bytes:
                ims = glob.glob(basefile+'*.png')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                tempfile = basefile + te
                if os.path.exists(tempfile):
                    os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, **kwargs):
    pngs = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return images


if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = r'\begin{equation}\mathcal{ L}\nonumber\end{equation}'

    print('Equation is: %s' % src)
    print(Latex(src).write())
