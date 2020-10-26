/**You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.


Input: l1 = [0], l2 = [0]
Output: [0]

**/


/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        if(l1 == NULL) return l2;
        if(l2 == NULL) return l1;
        
        ListNode* head=new ListNode();
        head->next=NULL;
        ListNode* temp=head;
        int sum=l1->val +l2->val;
        int carry=sum/10;
        temp->val=sum%10;
        l1=l1->next;
        l2=l2->next;
        
        while(l1!=NULL && l2!=NULL) {
            sum=carry + l1->val + l2->val;
            l1=l1->next; l2=l2->next;
            carry=sum/10;
            sum=sum%10;
            
            ListNode* t=new ListNode();
            t->val=sum;
            temp->next=t;
            temp=t;
            t->next=NULL;
        }
        
        while(l1!=NULL) {
            sum=carry+l1->val;
            l1=l1->next;
            carry=sum/10;
            ListNode* t=new ListNode();
            t->val=sum%10;
            temp->next=t;
            temp=t;
            t->next=NULL;
         }
         while(l2!=NULL) {
            sum=carry+l2->val;
            l2=l2->next;
            carry=sum/10;
            ListNode* t=new ListNode();
            t->val=sum%10;
            temp->next=t;
            temp=t;
            t->next=NULL;
         }
        
        if(carry!=0) {
            ListNode* t=new ListNode();
            t->val=carry;
            temp->next=t;
            temp=t;
            t->next=NULL;
        }
        return head;
        
            
        
    }
};
