#include <stddefcu.h>
#include <stringcu.h>
#include <grpcu.h>

__BEGIN_DECLS;

static __device__ group __grps[] = { { "std", 1, nullptr }, { nullptr } };
static __device__ group *__grpIdx = nullptr;

/* get group database entry for a group ID */
__device__ struct group *getgrgid_(gid_t gid)
{
	register group *p = __grps;
	while (p->gr_name && p->gr_gid != gid) p++;
	return (p->gr_name ? p : nullptr);
}

/* search group database for a name */
__device__ struct group *getgrnam_(const char *name)
{
	if (!name) return nullptr;
	register group *p = __grps;
	while (p->gr_name && strcmp(p->gr_name, name)) *p++;
	return (p->gr_name ? p : nullptr);
}

/* get the group database entry */
__device__ struct group *getgrent_()
{
	if (!__grpIdx) __grpIdx = __grps;
	else if (__grpIdx->gr_name) __grpIdx++;
	return (__grpIdx->gr_name ? __grpIdx : nullptr);
}

/* close the group database */
/* setgrent - reset group database to first entry */
__device__ void endgrent_()
{
	__grpIdx = nullptr;
}

__END_DECLS;
