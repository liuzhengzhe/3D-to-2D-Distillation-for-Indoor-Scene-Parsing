f=open('listtest')
fw=open('listtest2','w')
for line in f:
  fw.write(line.replace('/media/sda/lzz/tasks/','../data/'))
f.close()
fw.close()
