#include <stddefcu.h>
#include <grpcu.h>

__BEGIN_DECLS;

/* get group database entry for a group ID */
__device__ struct group *getgrgid_(gid_t gid)
{
	panic("Not Implemented");
	return nullptr;
}

/* search group database for a name */
__device__ struct group *getgrnam_(const char *name)
{
	panic("Not Implemented");
	return nullptr;
}

/* get the group database entry */
__device__ struct group *getgrent_()
{
	panic("Not Implemented");
	return nullptr;
}

/* close the group database */
__device__ void endgrent_()
{
	panic("Not Implemented");
}

/* reset group database to first entry */
__device__ void setgrent_()
{
	panic("Not Implemented");
}

__END_DECLS;
