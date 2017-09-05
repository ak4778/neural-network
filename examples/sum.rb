# Hello world regressor

require 'redis'

r = Redis.new
r.del(:sumnet)
#r.send('nr.create',:sumnet,:regressor,2,10,'->',1,:DATASET,1000,:TEST,500,:NORMALIZE)
r.send('nr.create',:sumnet,:regressor,2,3,'->',1,:DATASET,1000,:TEST,500,:NORMALIZE)

#1150.times {
#    a = rand(1000)
#    b = rand(1000)
#    r.send('nr.observe',:sumnet,a,b,'->',a*b)
#}
#
#1150.times {
#    a = rand(100)
#    b = rand(100)
#    r.send('nr.observe',:sumnet,a,b,'->',a*b)
#}
#
# Also train with smaller numbers, since the above training
# set will be unbalanced torward bigger numbers.
159.times {
    a = rand(15)
    b = rand(15)
    r.send('nr.observe',:sumnet,a,b,'->',a*b)
}

r.send('nr.train',:sumnet,:maxtime,2000)

sleep(3)

puts "50 + 100 = #{r.send('nr.run',:sumnet,50,100)}"
puts "20 + 40 = #{r.send('nr.run',:sumnet,20,40)}"
puts "2 + 4 = #{r.send('nr.run',:sumnet,2,4)}"
puts "5 + 2 = #{r.send('nr.run',:sumnet,5,2)}"
puts "7 + 9 = #{r.send('nr.run',:sumnet,7,9)}"
puts "1 + 4 = #{r.send('nr.run',:sumnet,1,4)}"
puts "3 + 6 = #{r.send('nr.run',:sumnet,3,6)}"
puts "3 + 4 = #{r.send('nr.run',:sumnet,3,4)}"
puts "7 + 5 = #{r.send('nr.run',:sumnet,7,5)}"
puts "4 + 5 = #{r.send('nr.run',:sumnet,4,5)}"
puts "5 + 5 = #{r.send('nr.run',:sumnet,3,5)}"
puts "9 + 2 = #{r.send('nr.run',:sumnet,9,2)}"
puts "6 + 8 = #{r.send('nr.run',:sumnet,6,8)}"
puts "111 + 890 = #{r.send('nr.run',:sumnet,111,890)}"
puts "4 + 23 = #{r.send('nr.run',:sumnet,4,23)}"
puts "98 + 657 = #{r.send('nr.run',:sumnet,98,657)}"
