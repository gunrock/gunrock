doxy-boot.js and customdoxygen.css appears to have come from:

https://github.com/Velron/doxygen-bootstrapped

customdoxygen had a bunch of font stuff though, which was not in the current build. So I patched that back in. Also changed some colors to make the header look decent (it was broken).

doxy-boot: copied some menu items (pages, files, globals, hierarchy)

Also patch into header.html:

https://github.com/Velron/doxygen-bootstrapped/blob/master/example-site/header.html

footer.html is identical.



Bootstrap: Replaced both bootstrap files with latest 3.x release from here:
http://getbootstrap.com/getting-started/#download
