// Original repository: https://github.com/tniessen/iperf-windows

#include <WinSock2.h>
#include <Windows.h>
#pragma comment(lib, "ws2_32.lib")

#ifndef UTSNAME_H
#define UTSNAME_H

#define UTSNAME_MAXLENGTH 256

struct utsname {
  char sysname[UTSNAME_MAXLENGTH];   // name of this implementation of the
                                     // operating system
  char nodename[UTSNAME_MAXLENGTH];  // name of this node within an
                                     // implementation - dependent
                                     // communications network
  char release[UTSNAME_MAXLENGTH];   //  current release level of this
                                     //  implementation
  char version[UTSNAME_MAXLENGTH];   //  current version level of this release
  char machine[UTSNAME_MAXLENGTH];   //  name of the hardware type on which the
                                     //  system is running
};

int uname(struct utsname* name);

#endif

int uname(struct utsname* name) {
  OSVERSIONINFO versionInfo;
  SYSTEM_INFO sysInfo;

  // Get Windows version info
  ZeroMemory(&versionInfo, sizeof(OSVERSIONINFO));
  versionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
  GetVersionEx(&versionInfo);

  // Get hardware info
  ZeroMemory(&sysInfo, sizeof(SYSTEM_INFO));
  GetSystemInfo(&sysInfo);

  // Set implementation name
  strcpy(name->sysname, "Windows");
  itoa(versionInfo.dwBuildNumber, name->release, 10);
  sprintf(name->version, "%i.%i", versionInfo.dwMajorVersion,
          versionInfo.dwMinorVersion);

  // Set hostname
  if (gethostname(name->nodename, UTSNAME_MAXLENGTH) != 0) {
    return WSAGetLastError();
  }

  // Set processor architecture
  switch (sysInfo.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64:
      strcpy(name->machine, "x86_64");
      break;
    case PROCESSOR_ARCHITECTURE_IA64:
      strcpy(name->machine, "ia64");
      break;
    case PROCESSOR_ARCHITECTURE_INTEL:
      strcpy(name->machine, "x86");
      break;
    case PROCESSOR_ARCHITECTURE_UNKNOWN:
    default:
      strcpy(name->machine, "unknown");
  }

  return 0;
}