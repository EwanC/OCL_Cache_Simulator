#Get file to plot as input argument
args <- commandArgs(trailingOnly = TRUE)

input =args[1] 

#Read file into table
file = read.table(input)

#Get the memory addresses
addr = (file[,1])
mem_addr = as.numeric(addr) - min(as.numeric(addr))
x = 1:length(addr)

#read or write
read= file[,2]

#get dimensions
D1 = file[,3]
D2 = file[,4]
D3 = file[,5]

#find max dimensions
max1 = max(D1) + 1
max2 = max(D2) + 1
max3 = max(D3) + 1

#open .png file to plot to
png(filename=paste(input,".png",sep=""),width=1680,height=1050,units="px")

#use red to black colour scheme
redblack = colorRampPalette(c('red','black'))
redblack =  redblack(max1*max2*max3)

#get unique thread id for each entry
ind = (D3* max1 * max2)+(D2 * max1) + D1 +1 

#make axis and title fonts larger
par(cex.axis=2, cex.lab=2, cex.main=2,mar=c(4,7,2,1))

#create plot  
plot(
	x,mem_addr
	,col= redblack[ind]
	,pch=read+15
	,ylab="Element"
    ,xlab="Execution Order"
    ,main="Scheduled Memory Accesses"
    ,yaxt="n"
 )

#add label to y axis
axis(2,mem_addr,label=mem_addr)

dev.off()
