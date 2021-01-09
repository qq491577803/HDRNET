# HDRNET
A detailed implement of Deep Bilateral Learning for Real-Time Image Enhancement
HDRNET tensorflow实现。文章的整体tensorflow框架基本搭建完善，但是在slicing那个地方，由于现有的tensorflow实现比较困难，使用numpy简单的实现了一下slicing操作
但是网络训练不起来。原文中是把slicing的过程编译成了SO文件，然后用tensorflow调用。
