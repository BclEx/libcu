/*
** Return the declared type of a column.  Or return zDflt if the column 
** has no declared type.
**
** The column type is an extra string stored after the zero-terminator on
** the column name if and only if the COLFLAG_HASTYPE flag is set.
*/
char *sqlite3ColumnType(Column *pCol, char *zDflt){
  if( (pCol->colFlags & COLFLAG_HASTYPE)==0 ) return zDflt;
  return pCol->zName + strlen(pCol->zName) + 1;
}