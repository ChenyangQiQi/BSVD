# This python file implement the idea of buffer as buffer conv
#%%
input_seq = [1,2,3,4,5]

def OpAdd(left, center, right):
    return left+center+right

class BufferAdd():
    def __init__(self) -> None:
        super(BufferAdd, self).__init__()
        self.op = OpAdd
        self.left = 0
        self.center = None
    def step(self, input_right):
        # order of input / center None need to think
        if self.center is None:
            self.center = input_right
            if input_right is not None:
                print("%d+none+%d = none"%(self.left, input_right))
            else:
                print("%d+none+none = none"%(self.left))
            # print("self.center is None")
            return None
        elif input_right is None:
            # print("input_right is None")
            if self.center == 0:
                print("%d+%d+none = 0"%(self.left, self.center))
                self.left = self.center
                self.center = input_right
                return 0
            print("%d+%d+none = 0"%(self.left, self.center))
            return 0

        else:
            output =  self.op(self.left, self.center, input_right)
            print("%d+%d+%d = %d"%(self.left, self.center, input_right, output))
            # if output == 57:
                # a = 1
            self.left = self.center
            self.center = input_right
            return output


class AddNet():
    def __init__(self) -> None:
        super(AddNet, self).__init__()
        self.add1 = BufferAdd()
        self.add2 = BufferAdd()
        # self.add3 = BufferAdd()
        # self.add4 = BufferAdd()

    def feedin_one_element(self, x):
        print("state after feed in ", x)
        x = self.add1.step(x)
        x = self.add2.step(x)
        
        self.print_state()
        # x = self.add3.step(x)
        # x = self.add4.step(x)
        return x
    def print_state(self):
        def print_buffer(buffer):
            # print("buffer left is", buffer.left)
            print("buffer center is", buffer.center)
        print("start of the first buffer")
        print_buffer(self.add1)
        print_buffer(self.add2)
        # print_buffer(self.add3)
        # print_buffer(self.add4)
    # def end(self):
        
    def forward(self, input_seq):
        out_seq = []
        for x in input_seq:
            # print("feed in %d"%x)
            out_seq.append(self.feedin_one_element(x))
        
        end_out = self.feedin_one_element(0)
        out_seq.append(end_out)
        # end_out = self.feedin_one_element(0)
        # end stage
        while 1:
            # print("feed in none")
            end_out = self.feedin_one_element(None)
            if end_out is None:
                break
            out_seq.append(end_out)
        return out_seq
#%%
add_net = AddNet()

out_seq = add_net.forward(input_seq)
print(out_seq)
# add_net.print_state()
            
        
# %%
