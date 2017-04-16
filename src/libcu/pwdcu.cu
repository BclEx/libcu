#include <stddefcu.h>
#include <stringcu.h>
#include <pwdcu.h>

__BEGIN_DECLS;

static __device__ passwd __pwds[] = { { "std", 1, 1 }, { nullptr } };
static __device__ passwd *__pwdIdx = nullptr;

/* search user database for a user ID */
__device__ struct passwd *getpwuid_(uid_t uid)
{
	register passwd *p = __pwds;
	while (p->pw_name && p->pw_uid != uid) *p++;
	return (p->pw_name ? p : nullptr);
}

/* search user database for a name */
__device__ struct passwd *getpwnam_(const char *name)
{
	if (!name) return nullptr;
	register passwd *p = __pwds;
	while (p->pw_name && strcmp(p->pw_name, name)) *p++;
	return (p->pw_name ? p : nullptr);
}

/* get user database entry [NOTSAFE] */
__device__ struct passwd *getpwent_()
{
	if (!__pwdIdx) __pwdIdx = __pwds;
	else if (__pwdIdx->pw_name) __pwdIdx++;
	return (__pwdIdx->pw_name ? __pwdIdx : nullptr);
}

/* close the user database */
/* setpwent - reset user database to first entry */
__device__ void endpwent_()
{
	__pwdIdx = nullptr;
}

__END_DECLS;
